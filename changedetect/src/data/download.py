"""
Utilities for downloading satellite imagery from various sources.
"""
import os
import sys
import subprocess
import logging
import requests
import json
import datetime
import time
from pathlib import Path
import argparse
from tqdm import tqdm
from sentinelsat import SentinelAPI, geojson_to_wkt

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_sentinel2_copernicus(output_dir, lat, lon, date_range, cloud_cover_max=10, 
                                api_url="https://catalogue.dataspace.copernicus.eu/odata/v1/"):
    """
    Download Sentinel-2 imagery from Copernicus Data Space Ecosystem.
    
    Args:
        output_dir (str): Directory to save downloaded imagery
        lat (float): Latitude of the point of interest
        lon (float): Longitude of the point of interest
        date_range (tuple): (start_date, end_date) in 'YYYY-MM-DD' format
        cloud_cover_max (int): Maximum cloud cover percentage
        api_url (str): URL of the Copernicus API
        
    Returns:
        list: Paths to downloaded files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse date range
    start_date, end_date = date_range
    
    logger.info(f"Searching for Sentinel-2 imagery at ({lat}, {lon}) from {start_date} to {end_date}")
    
    # Calculate the search area (0.1 degree buffer)
    search_box = {
        "west": lon - 0.1,
        "south": lat - 0.1,
        "east": lon + 0.1,
        "north": lat + 0.1
    }
    
    # Convert to WKT format for ODATA API
    wkt_polygon = f"POLYGON(({search_box['west']} {search_box['south']}, {search_box['west']} {search_box['north']}, " \
                 f"{search_box['east']} {search_box['north']}, {search_box['east']} {search_box['south']}, " \
                 f"{search_box['west']} {search_box['south']}))"
    
    # Prepare search parameters
    search_url = f"{api_url}Products"
    search_params = {
        "$filter": f"Collection/Name eq 'SENTINEL-2' "
                  f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_polygon}') "
                  f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
                  f"and ContentDate/Start lt {end_date}T23:59:59.999Z "
                  f"and Attributes/OData.CSC.DoubleAttribute/any(att:att/Name eq 'cloudCover' and att/Value le {cloud_cover_max})",
        "$top": 10,  # Limit the number of results
        "$expand": "Attributes"
    }
    
    # Check if COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET are available in environment
    client_id = os.environ.get('COPERNICUS_CLIENT_ID')
    client_secret = os.environ.get('COPERNICUS_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        logger.warning("COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET environment variables are required")
        logger.warning("Please set these variables with your Copernicus API credentials")
        logger.warning("Visit https://dataspace.copernicus.eu/userguide/Help-Center/Service-Terms-Conditions-Usage to create an account")
        return []
    
    try:
        # Get OAuth token
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        token_data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret
        }
        
        token_response = requests.post(token_url, data=token_data)
        token_response.raise_for_status()
        access_token = token_response.json().get('access_token')
        
        # Set authorization header
        headers = {'Authorization': f'Bearer {access_token}'}
        
        # Search for products
        response = requests.get(search_url, params=search_params, headers=headers)
        response.raise_for_status()
        
        # Process response
        search_results = response.json()
        products = search_results.get('value', [])
        
        if not products:
            logger.warning("No products found matching the criteria")
            return []
        
        logger.info(f"Found {len(products)} products")
        
        # List to store downloaded file paths
        downloaded_files = []
        
        # Download each product
        for product in products:
            product_id = product.get('Id')
            product_name = product.get('Name')
            
            logger.info(f"Downloading product: {product_name}")
            
            # Get product download URL
            download_url = f"{api_url}Products({product_id})/$value"
            
            # Output path
            output_path = os.path.join(output_dir, f"{product_name}.zip")
            
            # Download file
            with requests.get(download_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(output_path, 'wb') as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, desc=product_name
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Extract the ZIP file if it's not too large
            try:
                import zipfile
                if os.path.getsize(output_path) < 10 * 1024 * 1024 * 1024:  # < 10 GB
                    extract_dir = os.path.join(output_dir, product_name)
                    os.makedirs(extract_dir, exist_ok=True)
                    with zipfile.ZipFile(output_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    logger.info(f"Extracted {product_name} to {extract_dir}")
            except Exception as e:
                logger.warning(f"Failed to extract {output_path}: {e}")
            
            downloaded_files.append(output_path)
        
        return downloaded_files
    
    except Exception as e:
        logger.error(f"Error downloading Sentinel-2 imagery: {e}")
        logger.exception(e)
        return []

def download_resourcesat2_bhoonidhi(output_dir, lat, lon, date_range, 
                                   username=None, password=None):
    """
    Download ResourceSat-2 imagery from Bhoonidhi portal.
    
    Args:
        output_dir (str): Directory to save downloaded imagery
        lat (float): Latitude of the point of interest
        lon (float): Longitude of the point of interest
        date_range (tuple): (start_date, end_date) in 'YYYY-MM-DD' format
        username (str, optional): Bhoonidhi username
        password (str, optional): Bhoonidhi password
        
    Returns:
        list: Paths to downloaded files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse date range
    start_date, end_date = date_range
    
    logger.info(f"Searching for ResourceSat-2 imagery at ({lat}, {lon}) from {start_date} to {end_date}")
    
    # Get credentials from arguments or environment variables
    username = username or os.environ.get('BHOONIDHI_USERNAME')
    password = password or os.environ.get('BHOONIDHI_PASSWORD')
    
    if not username or not password:
        logger.warning("Bhoonidhi username and password are required")
        logger.warning("Please provide them as arguments or set BHOONIDHI_USERNAME and BHOONIDHI_PASSWORD environment variables")
        logger.warning("Visit https://bhoonidhi.nrsc.gov.in/bhoonidhi/home.html to create an account")
        return []
    
    # Base URLs for Bhoonidhi API
    base_url = "https://bhoonidhi.nrsc.gov.in/api"
    login_url = f"{base_url}/login"
    search_url = f"{base_url}/dataSearch"
    order_url = f"{base_url}/orderSave"
    download_url = f"{base_url}/downloadOrder"
    
    try:
        # Create a session to maintain cookies
        session = requests.Session()
        
        # Login to Bhoonidhi
        login_data = {
            "username": username,
            "password": password
        }
        
        login_response = session.post(login_url, json=login_data)
        login_response.raise_for_status()
        
        # Check if login was successful
        if "Invalid credentials" in login_response.text:
            logger.error("Invalid Bhoonidhi credentials")
            return []
        
        # Define search parameters for ResourceSat-2 LISS-IV
        # Buffering the search area by 0.1 degrees
        search_params = {
            "sensor": "LISS IV",
            "satellite": "RESOURCESAT-2",
            "startDate": start_date,
            "endDate": end_date,
            "areaType": "POINT",
            "area": f"{lon},{lat}",
            "buffer": 0.1  # 0.1 degree buffer around the point
        }
        
        search_response = session.post(search_url, json=search_params)
        search_response.raise_for_status()
        
        search_results = search_response.json()
        products = search_results.get("features", [])
        
        if not products:
            logger.warning("No ResourceSat-2 products found matching the criteria")
            return []
        
        logger.info(f"Found {len(products)} ResourceSat-2 products")
        
        # List to store downloaded file paths
        downloaded_files = []
        
        # Process each product
        for product in products:
            product_id = product.get("id")
            product_name = product.get("properties", {}).get("productName", f"product_{product_id}")
            
            # Place an order for the product
            order_data = {
                "productId": product_id,
                "downloadType": "DIRECT"  # Direct download
            }
            
            order_response = session.post(order_url, json=order_data)
            order_response.raise_for_status()
            
            order_result = order_response.json()
            order_id = order_result.get("orderId")
            
            if not order_id:
                logger.warning(f"Failed to place order for product {product_name}")
                continue
            
            logger.info(f"Successfully placed order {order_id} for product {product_name}")
            
            # Wait for order processing (this may take time in real scenario)
            # In a real implementation, you might want to poll the order status
            time.sleep(5)
            
            # Download the ordered product
            download_params = {
                "orderId": order_id
            }
            
            download_response = session.get(download_url, params=download_params, stream=True)
            download_response.raise_for_status()
            
            # Output path
            output_path = os.path.join(output_dir, f"{product_name}.zip")
            
            # Save the file
            total_size = int(download_response.headers.get("content-length", 0))
            
            with open(output_path, 'wb') as f, tqdm(
                total=total_size, unit='B', unit_scale=True, desc=product_name
            ) as pbar:
                for chunk in download_response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded {product_name} to {output_path}")
            downloaded_files.append(output_path)
            
            # Try to extract the zip file
            try:
                import zipfile
                extract_dir = os.path.join(output_dir, product_name)
                os.makedirs(extract_dir, exist_ok=True)
                
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                logger.info(f"Extracted {product_name} to {extract_dir}")
            except Exception as e:
                logger.warning(f"Failed to extract {output_path}: {e}")
        
        return downloaded_files
    
    except Exception as e:
        logger.error(f"Error downloading ResourceSat-2 imagery: {e}")
        logger.exception(e)
        return []

def download_sentinel1_copernicus(output_dir, lat, lon, date_range, 
                                polarization="VV,VH", api_url="https://catalogue.dataspace.copernicus.eu/odata/v1/"):
    """
    Download Sentinel-1 SAR imagery from Copernicus Data Space Ecosystem.
    Used in Stage-2/3 for SAR data processing.
    
    Args:
        output_dir (str): Directory to save downloaded imagery
        lat (float): Latitude of the point of interest
        lon (float): Longitude of the point of interest
        date_range (tuple): (start_date, end_date) in 'YYYY-MM-DD' format
        polarization (str): Polarization modes to download (comma-separated)
        api_url (str): URL of the Copernicus API
        
    Returns:
        list: Paths to downloaded files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Parse date range
    start_date, end_date = date_range
    
    logger.info(f"Searching for Sentinel-1 imagery at ({lat}, {lon}) from {start_date} to {end_date}")
    
    # Calculate the search area (0.1 degree buffer)
    search_box = {
        "west": lon - 0.1,
        "south": lat - 0.1,
        "east": lon + 0.1,
        "north": lat + 0.1
    }
    
    # Convert to WKT format for ODATA API
    wkt_polygon = f"POLYGON(({search_box['west']} {search_box['south']}, {search_box['west']} {search_box['north']}, " \
                 f"{search_box['east']} {search_box['north']}, {search_box['east']} {search_box['south']}, " \
                 f"{search_box['west']} {search_box['south']}))"
    
    # Create polarization filter
    pol_list = polarization.split(',')
    pol_filter = " or ".join([f"contains(Name, '{pol.strip()}')" for pol in pol_list])
    
    # Prepare search parameters for Sentinel-1
    search_url = f"{api_url}Products"
    search_params = {
        "$filter": f"Collection/Name eq 'SENTINEL-1' "
                  f"and OData.CSC.Intersects(area=geography'SRID=4326;{wkt_polygon}') "
                  f"and ContentDate/Start gt {start_date}T00:00:00.000Z "
                  f"and ContentDate/Start lt {end_date}T23:59:59.999Z "
                  f"and ({pol_filter})",
        "$top": 10,  # Limit the number of results
        "$expand": "Attributes"
    }
    
    # Check if COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET are available in environment
    client_id = os.environ.get('COPERNICUS_CLIENT_ID')
    client_secret = os.environ.get('COPERNICUS_CLIENT_SECRET')
    
    if not client_id or not client_secret:
        logger.warning("COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET environment variables are required")
        logger.warning("Please set these variables with your Copernicus API credentials")
        logger.warning("Visit https://dataspace.copernicus.eu/userguide/Help-Center/Service-Terms-Conditions-Usage to create an account")
        return []
    
    try:
        # Get OAuth token
        token_url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
        token_data = {
            'grant_type': 'client_credentials',
            'client_id': client_id,
            'client_secret': client_secret
        }
        
        token_response = requests.post(token_url, data=token_data)
        token_response.raise_for_status()
        access_token = token_response.json().get('access_token')
        
        # Set authorization header
        headers = {'Authorization': f'Bearer {access_token}'}
        
        # Search for products
        response = requests.get(search_url, params=search_params, headers=headers)
        response.raise_for_status()
        
        # Process response
        search_results = response.json()
        products = search_results.get('value', [])
        
        if not products:
            logger.warning("No Sentinel-1 products found matching the criteria")
            return []
        
        logger.info(f"Found {len(products)} Sentinel-1 products")
        
        # List to store downloaded file paths
        downloaded_files = []
        
        # Download each product
        for product in products:
            product_id = product.get('Id')
            product_name = product.get('Name')
            
            # Check if product has the requested polarization
            if not any(pol in product_name for pol in pol_list):
                logger.info(f"Skipping product {product_name} as it doesn't have requested polarization")
                continue
                
            logger.info(f"Downloading Sentinel-1 product: {product_name}")
            
            # Get product download URL
            download_url = f"{api_url}Products({product_id})/$value"
            
            # Output path
            output_path = os.path.join(output_dir, f"{product_name}.zip")
            
            # Download file
            with requests.get(download_url, headers=headers, stream=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                
                with open(output_path, 'wb') as f, tqdm(
                    total=total_size, unit='B', unit_scale=True, desc=product_name
                ) as pbar:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            # Process the Sentinel-1 data (GRD to calibrated, terrain-corrected)
            try:
                # Check if SNAP is installed for SAR preprocessing
                snap_installed = subprocess.call(['gpt', '-h'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0
                
                if snap_installed:
                    # Extract the ZIP file
                    import zipfile
                    extract_dir = os.path.join(output_dir, product_name)
                    os.makedirs(extract_dir, exist_ok=True)
                    
                    with zipfile.ZipFile(output_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    # Find the Sentinel-1 manifest file
                    manifest_path = os.path.join(extract_dir, "manifest.safe")
                    
                    if os.path.exists(manifest_path):
                        # Process with SNAP Graph Processing Tool (GPT)
                        processed_path = os.path.join(output_dir, f"{product_name}_processed.tif")
                        
                        # Create XML graph for processing
                        graph_path = os.path.join(output_dir, "sar_preprocessing.xml")
                        with open(graph_path, 'w') as f:
                            f.write("""<graph>
                              <node id="Read">
                                <operator>Read</operator>
                                <parameters>
                                  <file>${input}</file>
                                </parameters>
                              </node>
                              <node id="Apply-Orbit-File">
                                <operator>Apply-Orbit-File</operator>
                                <sources>
                                  <source>Read</source>
                                </sources>
                              </node>
                              <node id="Calibration">
                                <operator>Calibration</operator>
                                <sources>
                                  <source>Apply-Orbit-File</source>
                                </sources>
                                <parameters>
                                  <outputSigmaBand>true</outputSigmaBand>
                                  <outputImageScaleInDb>false</outputImageScaleInDb>
                                </parameters>
                              </node>
                              <node id="Terrain-Correction">
                                <operator>Terrain-Correction</operator>
                                <sources>
                                  <source>Calibration</source>
                                </sources>
                                <parameters>
                                  <demName>SRTM 1Sec HGT</demName>
                                  <pixelSpacingInMeter>10</pixelSpacingInMeter>
                                  <outputComplex>false</outputComplex>
                                </parameters>
                              </node>
                              <node id="Write">
                                <operator>Write</operator>
                                <sources>
                                  <source>Terrain-Correction</source>
                                </sources>
                                <parameters>
                                  <file>${output}</file>
                                  <formatName>GeoTIFF</formatName>
                                </parameters>
                              </node>
                            </graph>""")
                        
                        # Run GPT for preprocessing
                        subprocess.run([
                            'gpt', graph_path,
                            '-Pinput=' + manifest_path,
                            '-Poutput=' + processed_path
                        ], check=True)
                        
                        logger.info(f"Processed Sentinel-1 data saved to {processed_path}")
                        downloaded_files.append(processed_path)
                    else:
                        logger.warning(f"Could not find manifest.safe file in {extract_dir}")
                        downloaded_files.append(output_path)
                else:
                    logger.warning("SNAP GPT not found. Skipping SAR preprocessing.")
                    downloaded_files.append(output_path)
            except Exception as e:
                logger.warning(f"Error processing Sentinel-1 data: {e}")
                downloaded_files.append(output_path)
        
        return downloaded_files
    
    except Exception as e:
        logger.error(f"Error downloading Sentinel-1 imagery: {e}")
        logger.exception(e)
        return []

def download_sentinel2(output_dir, lat, lon, start_date, end_date, user, password, cloud_cover=10):
    # Connect to Copernicus Open Access Hub
    api = SentinelAPI(user, password, 'https://scihub.copernicus.eu/dhus')
    # Define area of interest as a point
    footprint = f'POINT({lon} {lat})'
    # Search for products
    products = api.query(
        footprint,
        date=(start_date, end_date),
        platformname='Sentinel-2',
        cloudcoverpercentage=(0, cloud_cover)
    )
    # Download all found products
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    api.download_all(products, directory_path=output_dir)

def list_sample_locations():
    """List sample locations from the PS-10 problem statement."""
    locations = [
        {"name": "Snow", "lat": 34.0531, "lon": 74.3909},
        {"name": "Plain", "lat": 13.3143, "lon": 77.6157},
        {"name": "Hill", "lat": 31.2834, "lon": 76.7904},
        {"name": "Desert", "lat": 26.9027, "lon": 70.9543},
        {"name": "Forest", "lat": 23.7380, "lon": 84.2129},
        {"name": "Urban", "lat": 28.1740, "lon": 77.6126}
    ]
    
    logger.info("Sample locations for different terrain types:")
    for loc in locations:
        logger.info(f"{loc['name']}: {loc['lat']}, {loc['lon']}")
    
    return locations

def main(args=None):
    """Command-line interface for downloading satellite imagery."""
    if args is None:
        args = sys.argv[1:]
    
    parser = argparse.ArgumentParser(description="Download satellite imagery")
    parser.add_argument("--source", choices=["sentinel2", "resourcesat2", "sentinel1"], 
                      required=True, help="Source of satellite imagery")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--lat", type=float, help="Latitude of point of interest")
    parser.add_argument("--lon", type=float, help="Longitude of point of interest")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--list-locations", action="store_true", 
                      help="List sample locations from the problem statement")
    parser.add_argument("--cloud-cover", type=int, default=10, 
                      help="Maximum cloud cover percentage (for Sentinel-2)")
    parser.add_argument("--username", help="Username for data portal")
    parser.add_argument("--password", help="Password for data portal")
    
    parsed_args = parser.parse_args(args)
    
    if parsed_args.list_locations:
        list_sample_locations()
        return 0
    
    if parsed_args.lat is None or parsed_args.lon is None:
        logger.error("Latitude and longitude are required")
        return 1
    
    if parsed_args.start_date is None or parsed_args.end_date is None:
        logger.error("Start date and end date are required")
        return 1
    
    date_range = (parsed_args.start_date, parsed_args.end_date)
    
    if parsed_args.source == "sentinel2":
        download_sentinel2_copernicus(
            parsed_args.output, 
            parsed_args.lat, 
            parsed_args.lon, 
            date_range, 
            cloud_cover_max=parsed_args.cloud_cover
        )
    elif parsed_args.source == "resourcesat2":
        download_resourcesat2_bhoonidhi(
            parsed_args.output, 
            parsed_args.lat, 
            parsed_args.lon, 
            date_range, 
            username=parsed_args.username, 
            password=parsed_args.password
        )
    elif parsed_args.source == "sentinel1":
        download_sentinel1_copernicus(
            parsed_args.output, 
            parsed_args.lat, 
            parsed_args.lon, 
            date_range
        )
    
    return 0

if __name__ == "__main__":
    sys.exit(main())