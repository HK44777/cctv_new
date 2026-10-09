from qdrant_client import QdrantClient
from collections import defaultdict

# 1. Connect directly to the local folder
client = QdrantClient(path="qdrant_db")
collection_name = "persons"

print("\n=== Qdrant Database Viewer ===")

# 2. Check if the collection exists
try:
    collections = client.get_collections().collections
    if not any(c.name == collection_name for c in collections):
        print("Collection 'persons' does not exist yet.")
        exit()
except Exception as e:
    print(f"Error connecting to DB: {e}")
    exit()

# 3. Get total number of tracked people
info = client.get_collection(collection_name)
print(f"Total Records in DB: {info.points_count}")

# 4. Fetch ALL points using scroll
all_records = []
offset = None
while True:
    records, offset = client.scroll(
        collection_name=collection_name,
        limit=250,
        offset=offset,
        with_payload=True,
        with_vectors=True  # Ensure we fetch vectors to check their presence
    )
    all_records.extend(records)
    if offset is None:
        break

# 5. Group by camera
camera_groups = defaultdict(list)
for record in all_records:
    cam = record.payload.get("camera_id", "Unknown")
    camera_groups[cam].append(record)

print(f"\n{'='*60}")
print("PER-CAMERA SUMMARY")
print(f"{'='*60}")
for cam in sorted(camera_groups.keys()):
    recs = camera_groups[cam]
    durations = []
    for r in recs:
        start = r.payload.get("timestamp", 0) or 0
        end = r.payload.get("end_time", 0) or 0
        durations.append(end - start)

    avg_dur = sum(durations) / len(durations) if durations else 0
    print(f"\n  🎥 {cam}:")
    print(f"    Records (persons tracked): {len(recs)}")
    print(f"    Avg track duration:        {avg_dur:.1f}s")
    print(f"    Min/Max duration:          {min(durations):.1f}s / {max(durations):.1f}s")

print(f"\n{'='*80}")
print("FULL RECORD LISTING")
print(f"{'='*80}")

for cam in sorted(camera_groups.keys()):
    print(f"\n--- Camera: {cam} ---")
    for record in sorted(camera_groups[cam], key=lambda r: r.payload.get("timestamp", 0) or 0):
        start = record.payload.get("timestamp", "N/A")
        end   = record.payload.get("end_time", "N/A")
        
        try:
            dur = end - start
            flag = " ⚠️ ZERO/BAD DURATION" if dur <= 0 else ""
            time_info = f"{start:.2f}s to {end:.2f}s (Duration: {dur:.2f}s){flag}"
        except Exception:
            time_info = f"{start} to {end}"

        print(f"\n  🆔 UUID: {record.id}")
        print(f"    🕒 Time: {time_info}")
        
        # Display all metadata except timestamp/end_time/camera_id which are already shown
        print(f"    📝 Additional Metadata:")
        for key, value in record.payload.items():
            if key not in ["timestamp", "end_time", "camera_id"]:
                print(f"       - {key}: {value}")
        
        # Check Vector Presence
        # In Qdrant named vectors, record.vector will be a dictionary
        vectors = record.vector if record.vector else {}
        
        app_status  = "✅ Vector Present" if "appearance" in vectors else "❌ None"
        face_status = "✅ Vector Present" if "face" in vectors else "❌ None"
        gait_status = "✅ Vector Present" if "gait" in vectors else "❌ None"
        
        print(f"    🧠 Vector Status:")
        print(f"       - Appearance : {app_status}")
        print(f"       - Face       : {face_status}")
        print(f"       - Gait       : {gait_status}")
print("\n")
