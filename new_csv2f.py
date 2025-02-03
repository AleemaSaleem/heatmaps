import firebase_admin
from firebase_admin import credentials, db, storage, firestore
import folium
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from shapely.geometry import Polygon, Point
from scipy.spatial import ConvexHull
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from io import BytesIO
import geopandas as gpd
from PIL import Image
import io
import cv2
from concurrent.futures import ThreadPoolExecutor
from firebase_init import get_firestore_client
# Firebase Configuration
# service_account = "crop2x.json"

# # Initialize Firebase Admin SDK
# cred = credentials.Certificate(service_account)
# firebase_admin.initialize_app(cred, {
#     'databaseURL': 'https://cropx-6f03a-default-rtdb.firebaseio.com',
#     'storageBucket': 'cropx-6f03a.appspot.com'
# })

# Initialize Firestore
# firestore_client = firestore.client()
# firestore_client = None

def upload_csv_to_firebase_and_store_url(csv_file_name, folder_name="uploads"):
    """
    Uploads a CSV file to Firebase Storage and stores its public URL in Firestore.

    Args:
        csv_file (str): The local path of the CSV file to upload.
        folder_name (str): The name of the folder in Firebase Storage where the file will be uploaded.
    """
    # Initialize storage client
    bucket = storage.bucket()

    # Set the target folder path in Firebase Storage
    file_name = csv_file.split("/")[-1]  # Get the CSV file name from the path
    blob = bucket.blob(f"{folder_name}/{file_name}")

    try:
        # Upload the file to Firebase Storage
        blob.upload_from_filename(csv_file_name)

        # Make the file publicly accessible
        blob.make_public()

        # Get the public URL of the file
        public_url = blob.public_url
        print(f"File '{file_name}' uploaded successfully and is publicly accessible at: {public_url}")

        # Store the public URL in Firestore
        store_url_in_firestore(file_name, public_url, folder_name)

    except Exception as e:
        print(f"Failed to upload file to Firebase Storage: {e}")

def store_url_in_firestore(file_name, public_url, folder_name="uploads"):
    """
    Store the public URL of the uploaded file in Firestore.

    Args:
        file_name (str): The name of the uploaded CSV file.
        public_url (str): The public URL of the uploaded file.
        folder_name (str): The folder name in Firebase Storage.
    """
    
    firestore_client = get_firestore_client()
    # Reference to Firestore collection
    urls_ref = firestore_client.collection('file_urls')  # You can change the collection name as needed

    # Add a new document with the file name and URL
    doc_ref = urls_ref.document(file_name)  # Using file name as document ID
    doc_ref.set({
        'file_name': file_name,
        'public_url': public_url,
        'folder_name': folder_name,
        'timestamp': firestore.SERVER_TIMESTAMP  # Adds a timestamp
    })
    print(f"Public URL for '{file_name}' stored in Firestore successfully.")


def fetch_data_from_csv(csv_file_name):
    """Fetch data from CSV file and return as a DataFrame."""
    return pd.read_csv(csv_file_name)

def fetch_timestamps(device, data):
    """Extract timestamps from data for a specific device."""
    # Assuming `data` has a `device_id` and `timestamp` column.
    device_data = data[data['device_id'] == device]
    print("device_id", data["device_id"])
    return device_data['timestamp'].tolist()

# Function to determine the color for phosphorus levels
def get_phosphorus_color(phosphorus):
    if 0 <= phosphorus <= 10.99:
        return "lightyellow"
    elif 11 <= phosphorus <= 20.99:
        return "lightblue"
    elif 21 <= phosphorus <= 40:
        return "blue"
    elif phosphorus > 40:
        return "darkblue"
    return "gray"

def get_nitrogen_color(nitrogen):
    if 0 <= nitrogen <= 10.99:
        return "lightyellow"
    elif 11 <= nitrogen <= 20.99:
        return "lightgreen"
    elif 21 <= nitrogen <= 40:
        return "green"
    elif nitrogen > 40:
        return "darkgreen"
    return "gray"  # Default for out-of-range values

def get_Conductivity_color(Conductivity):
    if 0 <= Conductivity <= 200:
        return "beige"
    elif 200.1<= Conductivity <= 404.0:
        return "purple"
    elif 405 <= Conductivity <= 800:
        return "orange"
    elif 801 <= Conductivity <= 1600:
        return "darkorange"
    elif Conductivity > 1600:
        return "red"
    return "gray"  # Default for out-of-range values

def get_ph_color(ph):
    if 0 <= ph <= 200:
        return "beige"
    elif 200.1<= ph <= 404.0:
        return "purple"
    elif 405 <= ph <= 800:
        return "orange"
    elif 801 <= ph <= 1600:
        return "darkorange"
    elif ph > 1600:
        return "red"
    return "gray"  # Default for out-of-range values

def get_moisture_color(moisture):
    if 0 < moisture < 15:
        return "lightcyan"
    elif 15 < moisture <= 30.99:
        return "cyan"
    elif 31 <= moisture <= 60.99:
        return "lightblue"
    elif 61 <= moisture <= 80.99:
        return "blue"
    elif 81 <= moisture <= 100:
        return "darkblue"
    return "gray"  # Default for out-of-range values

# Adjust function for Potassium color mapping
def get_potassium_color(k):
    if 0 <= k <= 52.9999999999999:
        return "white"  # Deficient
    elif 53 <= k <= 85:
        return "peachpuff"  # Low
    elif 86 <= k <= 120:
        return "orange"  # Optimum
    elif 121 <= k <= 155:
        return "red"  # High
    elif k > 155:
        return "darkred"  # Excessive
    return "gray"  # Default for out-of-range values

# Function to calculate distance between two coordinates (in meters)
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = np.radians(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

def decimal_to_dms(lat, lon):
    def to_dms(value):
        degrees = int(value)
        minutes = int((abs(value) - abs(degrees)) * 60)
        seconds = (abs(value) - abs(degrees) - minutes / 60) * 3600
        return degrees, minutes, seconds

    lat_deg, lat_min, lat_sec = to_dms(lat)
    lat_dir = 'N' if lat >= 0 else 'S'

    lon_deg, lon_min, lon_sec = to_dms(lon)
    lon_dir = 'E' if lon >= 0 else 'W'

    return f"{abs(lat_deg)}°{lat_min}'{lat_sec:.2f}\"{lat_dir}, {abs(lon_deg)}°{lon_min}'{lon_sec:.2f}\"{lon_dir}"


def process_attribute(data, device_id, date, attribute, color_function, distance_threshold):
    # try:
    # Ensure columns are numeric
    data['latitude'] = pd.to_numeric(data.get('latitude', pd.Series()), errors='coerce')
    data['longitude'] = pd.to_numeric(data.get('longitude', pd.Series()), errors='coerce')
    data[attribute] = pd.to_numeric(data.get(attribute, pd.Series()), errors='coerce')

    # Drop rows with missing or invalid data for the attribute
    data = data.dropna(subset=['latitude', 'longitude', attribute])

    # Calculate mean latitude and longitude
    mean_lat, mean_lon = data[['latitude', 'longitude']].mean()
    data['Distance'] = data.apply(
        lambda row: haversine(row['latitude'], row['longitude'], mean_lat, mean_lon), axis=1
    )

    # Filter rows based on the distance threshold
    filtered_data = data[data['Distance'] <= distance_threshold]

    # Prepare data for heatmap generation
    valid_data = filtered_data[(filtered_data['latitude'] != 0) & (filtered_data['longitude'] != 0)]
    points = valid_data[['latitude', 'longitude']].values
    values = valid_data[attribute].values

    if len(points) < 4:
        print(f"Skipping heatmap for {attribute}: Not enough valid points.")
        return

    # Generate a grid for interpolation
    grid_x, grid_y = np.mgrid[
        valid_data['latitude'].min():valid_data['latitude'].max():100j,
        valid_data['longitude'].min():valid_data['longitude'].max():100j,
    ]
    grid_z = griddata(points, values, (grid_x, grid_y), method='nearest')

    # Check if the points are not all the same
    lat_diff = np.ptp(points[:, 0])  # Range in latitude
    lon_diff = np.ptp(points[:, 1])  # Range in longitude

    if lat_diff < 1e-5 and lon_diff < 1e-5:
        print(f"Skipping heatmap generation for device {device_id} on {date}: Points are too similar.")
        return

    # Convex hull for boundary
    hull = ConvexHull(points)
    polygon = Polygon([points[vertex] for vertex in hull.vertices])
    hull_vertices = [points[vertex] for vertex in hull.vertices]
    # Add the red boundary for the convex hull
    hull_boundary = [[lat, lon] for lat, lon in hull_vertices]
    hull_boundary.append(hull_boundary[0])  # Close the polygon by adding the first vertex at the end

    # Create the polygon with the correct order (lon, lat)
    polygon_for_area = Polygon([(lon, lat) for lat, lon in hull_vertices])  # Ensure (lon, lat) order

    # Print converted coordinates of the polygon
    print("Converted Coordinates of the polygon (lon, lat):")
    for coord in hull_vertices:
        print(coord[1], coord[0])  # Print as (lon, lat)

    # Create grid for interpolation
    grid_lat = np.linspace(points[:, 0].min(), points[:, 0].max(), 100)
    grid_lon = np.linspace(points[:, 1].min(), points[:, 1].max(), 100)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    grid_points = np.vstack([grid_lat.ravel(), grid_lon.ravel()]).T

    # Filter grid points within convex hull
    grid_within = np.array([point for point in grid_points if polygon.contains(Point(point))])
    if grid_within.size == 0:
        print(f"No grid points within the convex hull for device {device_id} on {date}.")
        return

    grid_lat_within, grid_lon_within = grid_within[:, 0], grid_within[:, 1]
    
    grid = griddata(points, values, (grid_lat_within, grid_lon_within), method='nearest')

    # Generate heatmap
    # m = folium.Map(location=[points[:, 0].mean(), points[:, 1].mean()], zoom_start=17,tiles="OpenStreetMap")
    m = folium.Map(location=[points[:, 0].mean(), points[:, 1].mean()], zoom_start=16)
    # Add Google Satellite Maps as the basemap
    folium.TileLayer(
        tiles="https://{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google",
        name="Google Satellite",
        max_zoom=200,
        subdomains=["mt0", "mt1", "mt2", "mt3"],
    ).add_to(m)
    folium.PolyLine(
        locations=hull_boundary,
        color='red',
        weight=0,
        fill=True,
        opacity=0.7,
        tooltip="Convex Hull Boundary"
    ).add_to(m)
    color_func = color_functions[attribute]
    for lat, lon, phos in zip(grid_lat_within, grid_lon_within, grid):
        if not np.isnan(phos):
            color = color_func(phos)
            folium.CircleMarker(
                location=[lat, lon],
                weight=0,
                radius=5,
                color=color,
                fill=True,
                fill_opacity=0.7,
                tooltip=f"{attribute}: {phos:.2f}"
            ).add_to(m)

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(index=[0], geometry=[polygon_for_area], crs="EPSG:4326")

    # Convert to a suitable projected coordinate system (e.g., UTM)
    gdf = gdf.to_crs(epsg=32643)  # UTM zone 43N

    # Calculate the area in square meters
    area_square_meters = gdf.geometry.area[0]

    # Convert square meters to acres
    area_acres = area_square_meters / 4046.86

    print(f"Area of the polygon: {area_square_meters:.2f} square meters")
    print(f"Area of the polygon: {area_acres:.2f} acres")

    hull_folium_polygon = folium.Polygon(locations=hull_vertices, color="red", weight=2, fill=False)
    hull_folium_polygon.add_to(m)

    # Set the map's bounds to the convex hull
    m.fit_bounds(hull_folium_polygon.get_bounds())

    # Save heatmap to Firebase Storage
    html_data = m._repr_html_()
    html_io = io.BytesIO(html_data.encode('utf-8'))
    bucket = storage.bucket()
    html_blob = bucket.blob(f"heatmaps/{device_id}/{date}/{attribute}.html")
    html_blob.upload_from_file(html_io, content_type='text/html')
    html_blob.make_public()
    html_url = html_blob.public_url
    print(f"Heatmap HTML file uploaded to Firebase Storage at {html_url}")

    print(f"Heatmap for {attribute} uploaded to Firebase Storage.")
    # except Exception as e:
    #     print(f"Error processing attribute {attribute}: {e}")

        # Save a PNG screenshot of the HTML file using Selenium
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)


    try:
        # Render the HTML heatmap
        driver.get(html_url)
        time.sleep(0.5)  # Allow time for the map to fully load

        required_width = driver.execute_script("return document.body.scrollWidth;")
        required_height = driver.execute_script("return document.body.scrollHeight;")
        driver.set_window_size(required_width * 1, required_height * 1.8)

        png = driver.get_screenshot_as_png()
        # Load the PNG into Pillow for processing
        with Image.open(io.BytesIO(png)) as img:
            # Define the cropping area (keep the width and reduce the height)
            left = 0
            top = 0
            right = img.width
            bottom = min(int(img.height * 0.6), img.height)
            # bottom = int(img.height * 0.6)  # Adjust the cropping ratio as needed (e.g., 80% of the height)
            cropped_img = img.crop((left, top, right, bottom))

            # Convert cropped image back to bytes
            cropped_png_io = io.BytesIO()
            cropped_img.save(cropped_png_io, format="PNG")
            cropped_png_data = cropped_png_io.getvalue()
        # Upload the PNG file to Firebase Storage
        png_blob = bucket.blob(f"heatmaps/{device_id}/{date}/{attribute}.png")
        png_blob.upload_from_string(cropped_png_data, content_type="image/png")
        png_blob.make_public()
        png_url = png_blob.public_url
        print(f"Heatmap PNG screenshot uploaded to Firebase Storage at {png_url}")

        nparr = np.frombuffer(cropped_png_data, np.uint8)

        image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

        # Convert image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert the image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # lower_red1 = np.array([0, 50, 50])
        # upper_red1 = np.array([10, 255, 255])
        # lower_red2 = np.array([170, 50, 50])
        # upper_red2 = np.array([180, 255, 255])

        lower_red1 = np.array([0, 120, 120])  # Adjusted for solid red
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 120, 120])
        upper_red2 = np.array([180, 255, 255])

        # Mask for red color
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        # cv2.imwrite('red_mask.png', red_mask)
        # Find contours of the red boundary
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
        # cv2.imwrite('contours_visualized.png', image)  # Save the image with contours drawn
        if contours:
            # Sort contours by area and select the largest one
            largest_contour = max(contours, key=cv2.contourArea)
        else:
            print(f"No red boundary detected for {attribute}!")
            return  # Skip to the next attribute
        
        # Compute the convex hull of the largest contour
        hull = cv2.convexHull(largest_contour)
        # Create a mask for the area inside the boundary
        mask = np.zeros_like(gray)
        cv2.fillPoly(mask, [hull], 255)
        # cv2.imwrite('mask.png', mask)

        # Extract the region of interest using the convex hull mask
        result = cv2.bitwise_and(image, image, mask=mask)

        # Find the bounding rectangle of the convex hull
        x, y, w, h = cv2.boundingRect(hull)

        # Crop the image to the bounding rectangle
        cropped_image = result[y:y+h, x:x+w]
        # cv2.imwrite('cropped_result.png', cropped_image)
        # Create an alpha channel (transparency)
        alpha_channel = mask[y:y+h, x:x+w]

        # Merge the cropped BGR image with the alpha channel
        bgr_channels = cv2.split(cropped_image)[:3]  # Ignore existing alpha, if any
        transparent_image = cv2.merge((*bgr_channels, alpha_channel))

        # Save the processed image to a PNG in memory
        _, processed_png = cv2.imencode('.png', transparent_image)
        processed_png_data = processed_png.tobytes()

        # Upload the processed PNG to Firebase Storage
        processed_blob = bucket.blob(f"heatmaps/{device_id}/{date}/{attribute}_processed.png")
        processed_blob.upload_from_string(processed_png_data, content_type="image/png")
        processed_blob.make_public()

        # Get the public URL of the processed image
        processed_url = processed_blob.public_url
        print(f"Processed image uploaded to Firebase Storage for {attribute} at {processed_url}")
        
        # Delete the intermediate blobs
        try:
            html_blob.delete()
            print(f"Deleted HTML file from Firebase Storage: {html_blob.name}")

            png_blob.delete()
            print(f"Deleted PNG file from Firebase Storage: {png_blob.name}")

        except Exception as e:
            print(f"Error deleting blobs from Firebase Storage: {e}")

    finally:
        driver.quit()

    # Extract polygon coordinates directly as latitude and longitude
    polygon_coords = list(polygon.exterior.coords)

    # Log the coordinates for verification
    print("Coordinates of the polygon (lat, lon):")
    for coord in polygon_coords:
        print(coord)

    # Format coordinates as a list of dictionaries for storing in Firebase
    polygon_coordinates = [{"latitude": lat, "longitude": lon} for lon, lat in polygon_coords]
    
    firestore_client = get_firestore_client()
    # Reference to the Firestore collection for devices
    device_ref = firestore_client.collection("devices").document(device_id)

    # Add the date as a subfield under the device document
    device_ref.set({
        date: {
            "device_id": device_id,
            'date': date,
            # f"{attribute}_html_url": html_url,
            # "png_url": processed_url,
            'date': date,
            'area in sq meters': area_square_meters,
            'area in acres': area_acres,
            # "coordinates": dms_coordinates
            "coordinates": polygon_coordinates
        }
    }, merge=True)  # Use merge=True to avoid overwriting the existing document
    device_ref.update({
        f"{date}.png_url": firestore.ArrayUnion([processed_url])
    })


def create_heatmap(data, device_id, date, attributes, color_functions, distance_threshold=1000):
    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_attribute, data.copy(), str(device_id), date, attr, color_functions[attr], distance_threshold
            )
            for attr in attributes if attr in color_functions
        ]
        for future in futures:
            future.result()  # Wait for all threads to complete

def check_heatmaps_exist(device_id, date, attributes):
    """
    Check if heatmaps for all attributes already exist in Firebase Storage.
    Returns True if all heatmaps exist, False otherwise.
    """
    bucket_name = "cropx-6f03a.appspot.com"  # Replace with your Firebase Storage bucket name
    # client = storage.Client()
    bucket = storage.bucket(bucket_name)

    for attribute in attributes:
        blob_path = f"heatmaps/{device_id}/{date}/{attribute}_processed.png"
        blob = bucket.blob(blob_path)
        if not blob.exists():
            print(f"Heatmap for attribute {attribute} does not exist. Proceeding with generation.")
            return False
    print(f"Heatmaps for all attributes already exist for device {device_id} on {date}.")
    return True

devices = [ ]

csv_file ="sensor_data1.csv"

# upload_csv_to_firebase_and_store_url(csv_file, folder_name="csv_files")
data = fetch_data_from_csv(csv_file)

attributes = ['phosphor', 'conductivity', 'nitrogen', 'moisture', 'pH','potassium']
color_functions = {
    'phosphor': get_phosphorus_color,
    'conductivity': get_Conductivity_color,
    'nitrogen': get_nitrogen_color,
    'moisture' : get_moisture_color,
    'pH' : get_ph_color,
    'potassium' : get_potassium_color
}


# Function to process the device data and generate heatmap (dummy function)
def process_device_data(data, attributes, color_functions):
    devices = data['device_id'].unique()
    print(f"Unique device IDs: {devices} (type: {type(devices)})")
    result = []
    
    for device in devices:
        # Fetch timestamps for the device
        timestamps = fetch_timestamps(device, data)
        
        if not timestamps:
            print(f"No timestamps found for device {device}.")
            continue

        # Group timestamps by date
        date_groups = {}
        for timestamp in timestamps:
            date = "-".join(timestamp.split(" ")[0].split("-")[:3])  # Extract the 'YYYY-MM-DD' date part
            date_groups.setdefault(date, []).append(timestamp)

        # Create a heatmap for each date
        for date, ts_list in date_groups.items():
            # Check if heatmaps for all attributes already exist
            if check_heatmaps_exist(device, date, attributes):
                continue  # Skip heatmap generation if all heatmaps exist

            all_data = pd.concat([data[data['timestamp'] == ts] for ts in ts_list], ignore_index=True)
            print(f"Data for device {device} on {date}:\n", all_data)
            
            if not all_data.empty:
                # create_heatmap(all_data, device, date, attributes=attributes, color_functions=color_functions)
                create_heatmap(data, device, date, attributes, color_functions, distance_threshold=1000)
 
            else:
                print(f"No data found for device {device} on {date}.")

    return result

# process_device_data(data, attributes, color_functions)


