from fastapi import FastAPI, UploadFile, Form, HTTPException, File, Query
import os, cv2, numpy as np, face_recognition, face_recognition_models
from datetime import datetime
from ServerSQL import get_register_connection, get_attendance_connection, get_employee_info_connection

from zeroconf import Zeroconf, ServiceInfo
import socket
import netifaces
import traceback

app = FastAPI()
# face_recognition model directory
model_dir = os.path.join(os.path.dirname(face_recognition_models.__file__), "models")
os.environ["FACE_RECOGNITION_MODEL_LOCATION"] = model_dir

# Interpret match distance
def interpret_distance(distance):
    if distance <= 0.20:
        return "✔ Very strong match (likely same person)"
    elif distance <= 0.30:
        return "✔ Acceptable match (same person)"
    elif distance <= 0.45:
        return "⚠ Weak match (maybe same person)"
    else:
        return "❌ Not a match (different person)"

# Rotation logic
def load_image_with_rotation_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    for angle in [0, 90, 180, 270]:
        print(f"Trying rotation angle: {angle}")
        rotated = image
        if angle == 90:
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            rotated = cv2.rotate(image, cv2.ROTATE_180)
        elif angle == 270:
            rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            print(f"Face detected at angle {angle}")
            return encodings[0]
    print("No face encoding detected at any angle.")
    return None

@app.post("/register-face/")
async def register_face(
    name: str = Form(...),
    face_id: str = Form(...),  # employee_pin from Flutter
    image: UploadFile = File(...)
):
    if image.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG or PNG images are allowed.")
    image_bytes = await image.read()
    encoding = load_image_with_rotation_bytes(image_bytes)
    if encoding is None:
        raise HTTPException(status_code=400, detail="No clear face detected in the uploaded image.")
    os.makedirs("photos/registry", exist_ok=True)
    filename = f"{face_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    photo_path = os.path.join("photos/registry", filename)
    with open(photo_path, "wb") as f:
        f.write(image_bytes)
    try:
        conn = get_register_connection()
        print(" Connected to DB (REGISTER_EMP):", conn.getinfo(2))
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM EmployeeList WHERE employee_pin = ?", (face_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=400, detail=f"Face ID '{face_id}' does not match any employee_pin in EmployeeList.")
        real_face_id = row[0]
        cursor.execute("SELECT COUNT(*) FROM RegisteredFaces WHERE face_id = ?", (real_face_id,))
        if cursor.fetchone()[0] > 0:
            raise HTTPException(status_code=409, detail=f"Face ID '{face_id}' is already registered.")
        cursor.execute("INSERT INTO RegisteredFaces (face_id, name, face_encoding, photo_path) VALUES (?, ?, ?, ?)",
                       (real_face_id, name, encoding.tobytes(), photo_path))
        conn.commit()
        print(" Face inserted into REGISTER_EMP.")
    except Exception as e:
        conn.rollback()
        print("❌ INSERT FAILED:", e)
        raise HTTPException(status_code=500, detail=f"Database error during registration: {e}")
    finally:
        conn.close()
    try:
        conn_emp = get_employee_info_connection()
        cursor_emp = conn_emp.cursor()
        cursor_emp.execute("INSERT INTO EmployeeInfo (face_id, name) VALUES (?, ?)", (real_face_id, name))
        conn_emp.commit()
    except Exception as e:
        print(f"[WARN] Could not insert into EmployeeInfo: {e}")
    finally:
        conn_emp.close()
    return {
        "success": True,
        "message": "Face registered successfully.",
        "photo_path": photo_path
    }

@app.post("/recognize-face/")
async def recognize_face(
    image: UploadFile = File(...),
    face_id: str = Form(None),
    log_type: str = Form("IN")
):
    if image.content_type not in ["image/jpeg", "image/jpg", "image/png"]:
        raise HTTPException(status_code=400, detail="Only JPG or PNG images are allowed.")

    try:
        image_bytes = await image.read()
        encoding = load_image_with_rotation_bytes(image_bytes)
        if encoding is None:
            raise HTTPException(status_code=400, detail="No clear face detected in the uploaded image.")
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to process image or extract face encoding: {e}")

    THRESHOLD = 0.32
    real_face_id = None
    name = None
    distance = None

    try:
        conn = get_register_connection()
        cursor = conn.cursor()

        if face_id:
            cursor.execute("SELECT id FROM EmployeeList WHERE employee_pin = ?", (face_id,))
            emp = cursor.fetchone()
            if not emp:
                raise HTTPException(status_code=404, detail=f"Employee pin '{face_id}' not found in EmployeeList.")
            real_face_id = emp[0]
            cursor.execute("SELECT face_id, name, face_encoding FROM RegisteredFaces WHERE face_id = ?", (real_face_id,))
            row = cursor.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Face ID '{real_face_id}' not found in RegisteredFaces.")
            db_face_id, name, encoding_blob = row
            known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
            if known_encoding.shape != (128,):
                raise HTTPException(status_code=500, detail=f"Stored face encoding is malformed for Face ID {db_face_id}.")
            distance = face_recognition.face_distance([known_encoding], encoding)[0]
            print(f"[MATCH] Face ID '{real_face_id}' - Distance = {distance:.4f}")
            if distance > THRESHOLD:
                raise HTTPException(status_code=401, detail=f"Face mismatch. Distance = {distance:.4f}")
        else:
            cursor.execute("SELECT face_id, name, face_encoding FROM RegisteredFaces")
            rows = cursor.fetchall()
            best_match = None
            best_distance = 1.0
            for row in rows:
                db_face_id, temp_name, encoding_blob = row
                try:
                    known_encoding = np.frombuffer(encoding_blob, dtype=np.float64)
                    if known_encoding.shape != (128,):
                        print(f"[WARN] Skipping Face ID {db_face_id}: Invalid encoding shape {known_encoding.shape}")
                        continue
                    dist = face_recognition.face_distance([known_encoding], encoding)[0]
                    if dist < best_distance:
                        best_distance = dist
                        best_match = (db_face_id, temp_name)
                except Exception as e:
                    print(f"[ERROR] Failed comparing encoding for Face ID {db_face_id}: {e}")
                    traceback.print_exc()
            if best_match is None or best_distance > THRESHOLD:
                raise HTTPException(status_code=401, detail=f"No match found. Closest distance = {best_distance:.4f}")
            real_face_id, name = best_match
            distance = best_distance
        conn.close()
    except HTTPException:
        conn.close()
        raise
    except Exception as e:
        conn.close()
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error during face matching: {e}")

    try:
        os.makedirs("photos/logs", exist_ok=True)
        filename = f"{real_face_id}_{log_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        photo_path = os.path.join("photos/logs", filename)
        with open(photo_path, "wb") as f:
            f.write(image_bytes)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to save log image: {e}")

    try:
        conn_att = get_attendance_connection()
        cursor_att = conn_att.cursor()

        #  Fetch employee details from EmployeeList
        conn_info = get_register_connection()
        cursor_info = conn_info.cursor()
        cursor_info.execute("""
            SELECT employee_no, department, store_id, schedule_tag
            FROM EmployeeList
            WHERE id = ?
        """, (real_face_id,))
        emp_data = cursor_info.fetchone()
        conn_info.close()

        if not emp_data:
            raise HTTPException(status_code=500, detail="Failed to fetch employee info for logging.")

        employee_no, department, store_id, schedule_tag = emp_data

        # Insert into emp_attendance
        cursor_att.execute("""
            INSERT INTO dbo.emp_attendance (
                emp_id, att_date, in_out, filename, full_path, first_name,
                emp_no, on_duty_time, department, store_id, schedule_tag,
                captured_from
            )
            VALUES (?, GETDATE(), ?, ?, ?, ?, ?, GETDATE(), ?, ?, ?, ?)
        """, (
            real_face_id,                               # emp_id
            1 if log_type == "IN" else 0,               # in_out
            filename,                                   # filename
            photo_path,                                 # full_path
            name,                                       # first_name
            employee_no,                                # emp_no
            department,                                 # department
            store_id,                                   # store_id
            schedule_tag,                               # schedule_tag
            socket.gethostname()                        # captured_from
        ))

        conn_att.commit()
        conn_att.close()

    except Exception as e:
        traceback.print_exc()
        conn_att.rollback()
        conn_att.close()
        raise HTTPException(status_code=500, detail=f"Logging failed: {e}")

    return {
        "success": True,
        "message": f"{log_type} logged successfully.",
        "face_id": real_face_id,
        "name": name,
        "photo_path": photo_path,
        "distance": round(distance, 4),
        "match_strength": interpret_distance(distance)
    }



@app.get("/attendance-history/")
async def get_attendance_history(face_id: str = Query(None), name: str = Query(None)):
    conn = get_attendance_connection()
    cursor = conn.cursor()
    base_query = "SELECT face_id, name, log_type, timestamp, photo_path FROM AttendanceLogs"
    filters = []
    params = []
    if face_id:
        filters.append("face_id = ?")
        params.append(face_id)
    if name:
        filters.append("name = ?")
        params.append(name)
    if filters:
        base_query += " WHERE " + " AND ".join(filters)
    base_query += " ORDER BY timestamp DESC"
    cursor.execute(base_query, params)
    rows = cursor.fetchall()
    conn.close()
    return {
        "logs": [{
            "face_id": row[0],
            "name": row[1],
            "log_type": row[2],
            "timestamp": row[3].strftime("%Y-%m-%d %H:%M:%S"),
            "photo_path": row[4]
        } for row in rows]
    }

@app.get("/ping")
def ping():
    return {"status": "ok"}

def get_wifi_ip():
    for iface in netifaces.interfaces():
        addrs = netifaces.ifaddresses(iface).get(netifaces.AF_INET)
        if addrs:
            for addr in addrs:
                ip = addr.get("addr")
                if ip and not ip.startswith("127.") and not ip.startswith("169.") and not ip.startswith("172.") and not ip.startswith("192.168.56."):
                    return ip
    return "127.0.0.1"

def register_mdns_service():
    ip_address = get_wifi_ip()
    service_info = ServiceInfo(
        "_http._tcp.local.",
        "face-recognition-server._http._tcp.local.",
        addresses=[socket.inet_aton(ip_address)],
        port=8000,
        properties={},
        server="face.local.",
    )
    zeroconf = Zeroconf()
    zeroconf.register_service(service_info)
    print(f" Registering mDNS on {ip_address}:8000")
    return zeroconf

if __name__ == "__main__":
    import uvicorn
    mdns = register_mdns_service()
    try:
        uvicorn.run("ServerAPI:app", host="0.0.0.0", port=8000, reload=True)
    finally:
        mdns.close()