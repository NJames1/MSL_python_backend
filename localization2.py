"""
Cell Tower Fingerprinting + GPS Anchor Localization Pipeline
============================================================
Run this against your Render-hosted database to:
1. Parse all cell_data JSON blobs
2. Build cell tower pools per location
3. Verify proximity for non-GPS devices
4. Export enriched data for the dashboard
"""

import json
import re
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import psycopg2
import warnings
warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────
# 1. DATABASE CONNECTION
# ─────────────────────────────────────────────

def get_connection(database_url: str):
    """
    Connect to your Render-hosted PostgreSQL database.
    Pass your DATABASE_URL from Render dashboard.
    e.g. postgresql://user:password@host:port/dbname
    """
    return psycopg2.connect(database_url)


def load_scans(conn) -> pd.DataFrame:
    """Load all scans from the database into a DataFrame."""
    query = """
        SELECT 
            id,
            cell_data,
            wifi_data,
            gps_lat,
            gps_lon,
            user_name,
            location_id
            
        FROM raw_scans
        ORDER BY id
    """
    df = pd.read_sql(query, conn)
    print(f"✅ Loaded {len(df)} scans from database")
    return df


# ─────────────────────────────────────────────
# 2. CELL DATA PARSER
# ─────────────────────────────────────────────

def parse_cell_fingerprint(cell_data_raw) -> dict:
    """
    Parse the cell_data JSON blob into a structured dict.
    Handles both string and dict inputs.
    
    Returns:
        {
            'cells': [{'cid': int, 'rsrp': int, 'isRegistered': bool, 'tac': int}, ...],
            'serving_cid': int | None,
            'neighbor_cids': [int, ...]
        }
    """
    if cell_data_raw is None:
        return {'cells': [], 'serving_cid': None, 'neighbor_cids': []}
    
    try:
        # Handle string input
        if isinstance(cell_data_raw, str):
            # Sometimes the JSON has escape issues — try to clean
            cleaned = cell_data_raw.replace('\\"', '"')
            data = json.loads(cleaned)
        else:
            data = cell_data_raw

        fingerprint = data.get('fingerprint', data)
        cell_info_list = fingerprint.get('cellInfo', [])
        
        cells = []
        serving_cid = None
        neighbor_cids = []
        
        for cell in cell_info_list:
            cid = cell.get('cid')
            rsrp = cell.get('rsrp', -999)
            is_registered = cell.get('isRegistered', False)
            tac = cell.get('tac')
            
            if cid is None:
                continue
                
            entry = {
                'cid': int(cid),
                'rsrp': int(rsrp) if rsrp is not None else -999,
                'isRegistered': bool(is_registered),
                'tac': tac
            }
            cells.append(entry)
            
            if is_registered:
                serving_cid = int(cid)
            else:
                neighbor_cids.append(int(cid))
        
        return {
            'cells': cells,
            'serving_cid': serving_cid,
            'neighbor_cids': neighbor_cids
        }
        
    except Exception as e:
        return {'cells': [], 'serving_cid': None, 'neighbor_cids': [], 'error': str(e)}


def parse_wifi_fingerprint(wifi_data_raw) -> dict:
    """
    Parse wifi_data JSON blob.
    
    Returns:
        {
            'networks': [{'bssid': str, 'ssid': str, 'rssi': int}, ...],
            'strongest_bssid': str | None
        }
    """
    if wifi_data_raw is None or wifi_data_raw == 'None':
        return {'networks': [], 'strongest_bssid': None}
    
    try:
        if isinstance(wifi_data_raw, str):
            data = json.loads(wifi_data_raw)
        else:
            data = wifi_data_raw
        
        networks = []
        for net in data.get('wifiInfo', data if isinstance(data, list) else []):
            networks.append({
                'bssid': net.get('bssid', ''),
                'ssid': net.get('ssid', ''),
                'rssi': int(net.get('rssi', net.get('level', -999)))
            })
        
        # Sort by signal strength
        networks.sort(key=lambda x: x['rssi'], reverse=True)
        strongest = networks[0]['bssid'] if networks else None
        
        return {'networks': networks, 'strongest_bssid': strongest}
    
    except Exception:
        return {'networks': [], 'strongest_bssid': None}


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Parse all cell and wifi data in the dataframe."""
    print("🔄 Parsing cell fingerprints...")
    parsed_cells = df['cell_data'].apply(parse_cell_fingerprint)
    
    df['parsed_cells'] = parsed_cells.apply(lambda x: x.get('cells', []))
    df['serving_cid_parsed'] = parsed_cells.apply(lambda x: x.get('serving_cid'))
    df['neighbor_cids'] = parsed_cells.apply(lambda x: x.get('neighbor_cids', []))
    df['all_cids'] = parsed_cells.apply(
        lambda x: [c['cid'] for c in x.get('cells', [])]
    )
    df['num_visible_cells'] = df['all_cids'].apply(len)
    
    # Extract RSRP for serving cell
    def get_serving_rsrp(row):
        for cell in row['parsed_cells']:
            if cell.get('isRegistered'):
                return cell['rsrp']
        return None
    
    df['serving_rsrp'] = df.apply(get_serving_rsrp, axis=1)
    
    print("🔄 Parsing WiFi fingerprints...")
    parsed_wifi = df['wifi_data'].apply(parse_wifi_fingerprint)
    df['wifi_networks'] = parsed_wifi.apply(lambda x: x.get('networks', []))
    df['num_wifi_networks'] = df['wifi_networks'].apply(len)
    df['strongest_wifi_bssid'] = parsed_wifi.apply(lambda x: x.get('strongest_bssid'))
    
    # Has GPS flag
    df['has_gps'] = df['gps_lat'].notna() & df['gps_lon'].notna()
    
    print(f"✅ Enrichment complete. GPS anchors: {df['has_gps'].sum()} / {len(df)}")
    return df


# ─────────────────────────────────────────────
# 3. CELL POOL BUILDER
# ─────────────────────────────────────────────

class CellPoolBuilder:
    """
    Builds and manages cell tower pools per location.
    A pool = set of CIDs observed at a location from GPS-anchored scans.
    """
    
    def __init__(self, min_anchor_scans: int = 2, min_cell_frequency: float = 0.1):
        """
        min_anchor_scans: minimum GPS-anchored scans needed to trust a pool
        min_cell_frequency: a CID must appear in at least X% of scans to be in pool
        """
        self.min_anchor_scans = min_anchor_scans
        self.min_cell_frequency = min_cell_frequency
        self.pools = {}           # location_id -> set of CIDs
        self.pool_stats = {}      # location_id -> stats dict
        self.cid_locations = {}   # cid -> list of location_ids (reverse index)
    
    def build(self, df: pd.DataFrame):
        """Build pools from GPS-anchored scans only."""
        anchor_df = df[df['has_gps'] == True].copy()
        
        print(f"🏗️  Building pools from {len(anchor_df)} GPS-anchored scans...")
        
        location_cid_counts = defaultdict(lambda: defaultdict(int))
        location_scan_counts = defaultdict(int)
        location_coords = defaultdict(list)
        
        for _, row in anchor_df.iterrows():
            loc = row['location_id']
            if not loc:
                continue
            
            location_scan_counts[loc] += 1
            location_coords[loc].append((row['gps_lat'], row['gps_lon']))
            
            for cid in row['all_cids']:
                location_cid_counts[loc][cid] += 1
        
        for loc, cid_counts in location_cid_counts.items():
            total_scans = location_scan_counts[loc]
            
            # Only include CIDs that appear frequently enough
            pool = set()
            cid_freqs = {}
            for cid, count in cid_counts.items():
                freq = count / total_scans
                cid_freqs[cid] = freq
                if freq >= self.min_cell_frequency:
                    pool.add(cid)
            
            coords = location_coords[loc]
            avg_lat = np.mean([c[0] for c in coords])
            avg_lon = np.mean([c[1] for c in coords])
            
            self.pools[loc] = pool
            self.pool_stats[loc] = {
                'location_id': loc,
                'pool_size': len(pool),
                'anchor_scan_count': total_scans,
                'avg_lat': avg_lat,
                'avg_lon': avg_lon,
                'cid_frequencies': cid_freqs,
                'is_reliable': total_scans >= self.min_anchor_scans
            }
        
        # Build reverse index
        for loc, pool in self.pools.items():
            for cid in pool:
                if cid not in self.cid_locations:
                    self.cid_locations[cid] = []
                self.cid_locations[cid].append(loc)
        
        print(f"✅ Built pools for {len(self.pools)} locations")
        for loc, stats in self.pool_stats.items():
            print(f"   📍 {loc}: {stats['pool_size']} CIDs from {stats['anchor_scan_count']} anchors")
        
        return self
    
    def verify_proximity(self, device_cids: list, top_k: int = 3) -> list:
        """
        Given a device's visible CIDs, find which location pools it matches.
        Returns ranked list of (location_id, confidence, matched_cids).
        """
        device_set = set(device_cids)
        results = []
        
        for loc, pool in self.pools.items():
            if not pool:
                continue
            
            overlap = device_set & pool
            
            # Jaccard-like score weighted by pool size
            precision = len(overlap) / len(device_set) if device_set else 0
            recall = len(overlap) / len(pool) if pool else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            if overlap:
                results.append({
                    'location_id': loc,
                    'confidence': round(f1, 3),
                    'overlap_count': len(overlap),
                    'matched_cids': list(overlap),
                    'pool_size': len(pool),
                    'in_proximity': f1 >= 0.25
                })
        
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:top_k]
    
    def get_summary(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.pool_stats.values()))


# ─────────────────────────────────────────────
# 4. PROXIMITY VERIFIER (apply to full dataset)
# ─────────────────────────────────────────────

def run_proximity_verification(df: pd.DataFrame, pool_builder: CellPoolBuilder) -> pd.DataFrame:
    """
    For every scan (especially non-GPS ones), run proximity verification
    and store the best-match location + confidence.
    """
    print("🔍 Running proximity verification on all scans...")
    
    results = []
    for _, row in df.iterrows():
        matches = pool_builder.verify_proximity(row['all_cids'])
        
        if matches:
            best = matches[0]
            results.append({
                'best_match_location': best['location_id'],
                'proximity_confidence': best['confidence'],
                'proximity_verified_auto': best['in_proximity'],
                'all_matches': json.dumps(matches)
            })
        else:
            results.append({
                'best_match_location': None,
                'proximity_confidence': 0.0,
                'proximity_verified_auto': False,
                'all_matches': '[]'
            })
    
    result_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, result_df], axis=1)
    
    verified = df['proximity_verified_auto'].sum()
    print(f"✅ Proximity verified: {verified}/{len(df)} scans ({100*verified/len(df):.1f}%)")
    return df


# ─────────────────────────────────────────────
# 5. ML FEATURE EXTRACTION + TRAINING
# ─────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame, top_cids: list = None) -> tuple:
    """
    Build feature matrix for ML classification.
    Features per scan:
    - RSRP for top N most common CIDs (0 if not seen)
    - Number of visible cells
    - Number of WiFi networks
    - Serving cell RSRP
    """
    if top_cids is None:
        # Find the most common CIDs across all scans
        all_cids_flat = [cid for cids in df['all_cids'] for cid in cids]
        cid_series = pd.Series(all_cids_flat)
        top_cids = cid_series.value_counts().head(50).index.tolist()
    
    rows = []
    for _, row in df.iterrows():
        feat = {}
        
        # CID RSRP features
        cid_rsrp_map = {c['cid']: c['rsrp'] for c in row['parsed_cells']}
        for cid in top_cids:
            feat[f'rsrp_{cid}'] = cid_rsrp_map.get(cid, -120)  # -120 = not visible
        
        # Aggregate features
        feat['num_visible_cells'] = row['num_visible_cells']
        feat['serving_rsrp'] = row['serving_rsrp'] if row['serving_rsrp'] else -120
        feat['num_wifi_networks'] = row['num_wifi_networks']
        
        rows.append(feat)
    
    feature_df = pd.DataFrame(rows)
    return feature_df, top_cids


class LocalizationModel:
    """Ensemble of RF + KNN for location classification."""
    
    def __init__(self):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
        self.label_encoder = LabelEncoder()
        self.top_cids = None
        self.is_trained = False
    
    def train(self, df: pd.DataFrame):
        """Train on GPS-anchored scans with known location_id."""
        labelled = df[df['has_gps'] & df['location_id'].notna()].copy()
        
        if len(labelled) < 10:
            print("⚠️  Not enough labelled data for ML training (need 10+ GPS scans)")
            return self
        
        print(f"🤖 Training ML models on {len(labelled)} labelled scans...")
        
        X, self.top_cids = build_feature_matrix(labelled)
        y = self.label_encoder.fit_transform(labelled['location_id'])
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        self.rf.fit(X_train, y_train)
        self.knn.fit(X_train, y_train)
        
        # Evaluate
        rf_preds = self.rf.predict(X_test)
        print("\n📊 Random Forest Results:")
        print(classification_report(
            y_test, rf_preds,
            target_names=self.label_encoder.classes_
        ))
        
        self.is_trained = True
        return self
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict location for all scans."""
        if not self.is_trained:
            print("⚠️  Model not trained yet")
            return df
        
        X, _ = build_feature_matrix(df, top_cids=self.top_cids)
        
        rf_probs = self.rf.predict_proba(X)
        rf_preds = self.rf.predict(X)
        
        knn_preds = self.knn.predict(X)
        
        df['ml_predicted_location'] = self.label_encoder.inverse_transform(rf_preds)
        df['ml_confidence'] = rf_probs.max(axis=1).round(3)
        df['knn_predicted_location'] = self.label_encoder.inverse_transform(knn_preds)
        
        # Ensemble: agree = higher confidence
        df['ml_ensemble_agree'] = df['ml_predicted_location'] == df['knn_predicted_location']
        
        return df
    
    def get_feature_importance(self) -> pd.DataFrame:
        if not self.is_trained:
            return pd.DataFrame()
        
        X_dummy, _ = build_feature_matrix(pd.DataFrame(), top_cids=self.top_cids)
        feature_names = [f'rsrp_{cid}' for cid in self.top_cids] + [
            'num_visible_cells', 'serving_rsrp', 'num_wifi_networks'
        ]
        
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(self.rf.feature_importances_)],
            'importance': self.rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


# ─────────────────────────────────────────────
# 6. EXPORT FOR DASHBOARD
# ─────────────────────────────────────────────

def export_dashboard_data(df: pd.DataFrame, pool_builder: CellPoolBuilder, output_dir: str = '.'):
    """Export all processed data as JSON files for the dashboard."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Pool summary
    pool_summary = pool_builder.get_summary()
    pool_summary.to_json(f'{output_dir}/pool_summary.json', orient='records')
    
    # 2. Per-location pool details (CIDs + frequencies)
    pool_details = {}
    for loc, stats in pool_builder.pool_stats.items():
        pool_details[loc] = {
            'cids': list(pool_builder.pools[loc]),
            'cid_frequencies': {str(k): v for k, v in stats['cid_frequencies'].items()},
            'avg_lat': stats['avg_lat'],
            'avg_lon': stats['avg_lon'],
            'anchor_count': stats['anchor_scan_count']
        }
    with open(f'{output_dir}/pool_details.json', 'w') as f:
        json.dump(pool_details, f)
    
    # 3. Scan summary (drop heavy columns)
    export_cols = [
        'id', 'user_name', 'location_id', 'has_gps', 'gps_lat', 'gps_lon',
        'serving_cid_parsed', 'serving_rsrp', 'num_visible_cells', 'num_wifi_networks',
        'best_match_location', 'proximity_confidence', 'proximity_verified_auto'
    ]
    available_cols = [c for c in export_cols if c in df.columns]
    scan_summary = df[available_cols].copy()
    scan_summary.to_json(f'{output_dir}/scan_summary.json', orient='records')
    
    # 4. Confidence histogram data
    if 'proximity_confidence' in df.columns:
        hist_data = df['proximity_confidence'].dropna().tolist()
        with open(f'{output_dir}/confidence_histogram.json', 'w') as f:
            json.dump(hist_data, f)
    
    print(f"✅ Dashboard data exported to {output_dir}/")


# ─────────────────────────────────────────────
# 7. MAIN PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(database_url: str, export_dir: str = './dashboard_data'):
    """
    Full pipeline:
    1. Load from DB
    2. Parse & enrich
    3. Build cell pools
    4. Run proximity verification
    5. Train ML model
    6. Export for dashboard
    """
    print("=" * 60)
    print("  CELL TOWER LOCALIZATION PIPELINE")
    print("=" * 60)
    
    # Step 1: Load
    conn = get_connection(database_url)
    df = load_scans(conn)
    conn.close()
    
    # Step 2: Parse & enrich
    df = enrich_dataframe(df)
    
    # Step 3: Build pools
    pool_builder = CellPoolBuilder(min_anchor_scans=2, min_cell_frequency=0.1)
    pool_builder.build(df)
    
    # Step 4: Proximity verification
    df = run_proximity_verification(df, pool_builder)
    
    # Step 5: ML
    model = LocalizationModel()
    model.train(df)
    if model.is_trained:
        df = model.predict(df)
    
    # Step 6: Export
    export_dashboard_data(df, pool_builder, output_dir=export_dir)
    
    print("\n🎉 Pipeline complete!")
    return df, pool_builder, model


# ─────────────────────────────────────────────
# USAGE EXAMPLE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Replace with your Render database URL
    DATABASE_URL = "postgresql://james:YmeXArVGRY19ermS7lXy1Op4fVv00Yro@dpg-d7n6k868bjmc738msds0-a.oregon-postgres.render.com/msl_live_demo"
    
    df, pools, model = run_pipeline(DATABASE_URL, export_dir='./dashboard_data')
    
    # Quick check
    print("\nSample proximity check:")
    test_cids = [2631691, 2631690, 2631680]
    matches = pools.verify_proximity(test_cids)
    for m in matches:
        print(f"  → {m['location_id']}: {m['confidence']:.0%} confidence ({m['overlap_count']} CIDs matched)")