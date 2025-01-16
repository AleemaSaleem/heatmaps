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
# Firebase Configuration
service_account = "crop2x.json"

# Initialize Firebase Admin SDK
cred = credentials.Certificate(service_account)
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://cropx-6f03a-default-rtdb.firebaseio.com',
    'storageBucket': 'cropx-6f03a.appspot.com'
})

# Initialize Firestore
firestore_client = firestore.client()

# Function to fetch all timestamps for a given device
def fetch_timestamps(device_id):
    path = f'/realtimedevices/{device_id}'
    ref = db.reference(path)
    timestamps = ref.get()
    if timestamps:
        return list(timestamps.keys())
    return []

# Function to fetch data from Firebase for a specific device and timestamp
def fetch_data(device_id, timestamp):
    path = f'/realtimedevices/{device_id}/{timestamp}'
    ref = db.reference(path)
    data = ref.get()
    if data:
        if isinstance(data, dict):
            data = [data]
        return pd.DataFrame(data)
    return pd.DataFrame()

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

# Function to create a heatmap for a given dataset
def create_heatmap(data, device_id, date, attributes, color_functions, distance_threshold=1000):
    # # Ensure columns are numeric
    
    for attribute in attributes:
        if attribute not in data.columns or attribute not in color_functions:
            print(f"Skipping heatmap for {attribute}: No data or color function available.")
            continue
        
        data['latitude'] = pd.to_numeric(data.get('latitude', pd.Series()), errors='coerce')
        data['longitude'] = pd.to_numeric(data.get('longitude', pd.Series()), errors='coerce')

        # Convert attribute data to numeric
        data[attribute] = pd.to_numeric(data.get(attribute, pd.Series()), errors='coerce')

        # Drop rows with missing or invalid data for the attribute
        data = data.dropna(subset=['latitude', 'longitude', attribute])
        print(f"Data after dropping invalid rows for {attribute}:", data)
        print(data[['latitude', 'longitude']].iloc[:30])


        # Calculate mean latitude and longitude
        mean_lat, mean_lon = data[['latitude', 'longitude']].mean()
        data['Distance'] = data.apply(
            lambda row: haversine(row['latitude'], row['longitude'], mean_lat, mean_lon), axis=1
        )

        # Filter rows based on the distance threshold
        filtered_data = data[data['Distance'] <= distance_threshold]
        print(f"Filtered data size for {attribute}: {len(filtered_data)} rows (from original {len(data)} rows)")

        # Prepare data for heatmap generation
        valid_data = filtered_data[(filtered_data['latitude'] != 0) & (filtered_data['longitude'] != 0)]
        points = valid_data[['latitude', 'longitude']].values
        values = valid_data[attribute].values

        if len(points) < 3:
            print(f"Skipping heatmap for {attribute}: Not enough valid points.")
            continue

        # Generate a grid for interpolation
        grid_x, grid_y = np.mgrid[
            valid_data['latitude'].min():valid_data['latitude'].max():100j,
            valid_data['longitude'].min():valid_data['longitude'].max():100j,
        ]
        grid_z = griddata(points, values, (grid_x, grid_y), method='nearest')


        # Check if the points are not all the same (if the difference in lat/lon is too small)
        lat_diff = np.ptp(points[:, 0])  # Peak-to-peak (range) in latitude
        lon_diff = np.ptp(points[:, 1])  # Peak-to-peak (range) in longitude

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
        m = folium.Map(location=[points[:, 0].mean(), points[:, 1].mean()], zoom_start=17,tiles="OpenStreetMap")
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


        # Save HTML to memory (instead of writing to a local file)
        html_data = m._repr_html_()  # Get the HTML representation of the map

        # Create in-memory file-like object for HTML
        html_io = BytesIO(html_data.encode('utf-8'))

        # Upload the HTML file to Firebase Storage
        bucket = storage.bucket()
        html_blob = bucket.blob(f"heatmaps/{device_id}/{date}/{attribute}.html")
        html_blob.upload_from_file(html_io, content_type='text/html')

        # Make the file publicly accessible
        html_blob.make_public()

        # Get the public URL of the uploaded HTML file
        html_url = html_blob.public_url
        print(f"Heatmap HTML file uploaded to Firebase Storage at {html_url}")

        # Save a PNG screenshot of the HTML file using Selenium
        options = Options()
        options.headless = True
        driver = webdriver.Chrome(options=options)


        try:
            # Render the HTML heatmap
            driver.get(html_url)
            time.sleep(3)  # Allow time for the map to fully load

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
                bottom = int(img.height * 0.6)  # Adjust the cropping ratio as needed (e.g., 80% of the height)
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

            lower_red1 = np.array([0, 120, 120])  # Adjusted for solid red
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 120, 120])
            upper_red2 = np.array([180, 255, 255])            

            # Mask for red color
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)

            # Find contours of the red boundary
            contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Sort contours by area and select the largest one
                largest_contour = max(contours, key=cv2.contourArea)
            else:
                print(f"No red boundary detected for {attribute}!")
                continue  # Skip to the next attribute

            # Compute the convex hull of the largest contour
            hull = cv2.convexHull(largest_contour)
            # Create a mask for the area inside the boundary
            mask = np.zeros_like(gray)
            cv2.fillPoly(mask, [hull], 255)

            # Extract the region of interest using the convex hull mask
            result = cv2.bitwise_and(image, image, mask=mask)

            # Find the bounding rectangle of the convex hull
            x, y, w, h = cv2.boundingRect(hull)

            # Crop the image to the bounding rectangle
            cropped_image = result[y:y+h, x:x+w]

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

        # Reference to the Firestore collection for devices
        device_ref = firestore_client.collection("devices").document(device_id)

        # Add the date as a subfield under the device document
        device_ref.set({
            date: {
                "device_id": device_id,
                'date': date,
                f"{attribute}_html_url": html_url,
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


attributes = ['phosphor', 'conductivity', 'nitrogen', 'moisture', 'pH','potassium']
color_functions = {
    'phosphor': get_phosphorus_color,
    'conductivity': get_Conductivity_color,
    'nitrogen': get_nitrogen_color,
    'moisture' : get_moisture_color,
    'pH' : get_ph_color,
    'potassium' : get_potassium_color
}

# # Prompt user for device IDs (comma-separated)
# device_input = input("Enter device IDs (comma-separated): ")
# devices = [device.strip() for device in device_input.split(",")]

devices=["2407050002"]
# devices=[]

for device in devices:
    timestamps = fetch_timestamps(device)
    if not timestamps:
        print(f"No timestamps found for device {device}.")
        continue

    # Group timestamps by date
    date_groups = {}
    for timestamp in timestamps:
        date = "-".join(timestamp.split("-")[:3])  # Assuming 'YYYY-MM-DD HH-MM-SS' format
        date_groups.setdefault(date, []).append(timestamp)

    # Create a heatmap for each date
    for date, ts_list in date_groups.items():
        all_data = pd.concat([fetch_data(device, ts) for ts in ts_list], ignore_index=True)
        print(f"Data for device {device} on {date}:\n", all_data)
        if not all_data.empty:
            create_heatmap(all_data, device, date, attributes=attributes, color_functions=color_functions)
        else:
            print(f"No data found for device {device} on {date}.")

# def get_device_ids():
#     # Prompt user for device IDs (comma-separated)
#     device_input = input("Enter device IDs (comma-separated): ")
#     devices = [device.strip() for device in device_input.split(",")]
#     return devices

# def process_device_data(devices, attributes, color_functions):
#     for device in devices:
#         timestamps = fetch_timestamps(device)
#         if not timestamps:
#             print(f"No timestamps found for device {device}.")
#             continue

#         # Group timestamps by date
#         date_groups = {}
#         for timestamp in timestamps:
#             date = "-".join(timestamp.split("-")[:3])  # Assuming 'YYYY-MM-DD HH-MM-SS' format
#             date_groups.setdefault(date, []).append(timestamp)

#         # Create a heatmap for each date
#         for date, ts_list in date_groups.items():
#             all_data = pd.concat([fetch_data(device, ts) for ts in ts_list], ignore_index=True)
#             print(f"Data for device {device} on {date}:\n", all_data)
#             if not all_data.empty:
#                 create_heatmap(all_data, device, date, attributes=attributes, color_functions=color_functions)
#             else:
#                 print(f"No data found for device {device} on {date}.")

# # # Main part of the code
# # devices = get_device_ids()
# process_device_data(devices, attributes, color_functions)
