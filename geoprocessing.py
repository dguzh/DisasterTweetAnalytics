import warnings
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from pyproj import CRS
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.preprocessing import OneHotEncoder

class Disaster:
    """
    A class to represent a disaster event with relevant information and methods.
    
    The class stores information about the disaster, including its subtype, location,
    and start date, and provides methods to process and obtain features for further analysis.
    
    Attributes:
        disaster_subtype (str): The disaster subtype.
        latitude (float): The latitude of the disaster event.
        longitude (float): The longitude of the disaster event.
        start_date (datetime): The start date of the disaster event.
        xyz_coords (tuple): The 3D coordinates (x, y, z) of the disaster event.
        one_hot_subtype (list): A one-hot-encoded list representing the disaster subtype.
    """

    def __init__(self, data):
        """
        Initializes a Disaster object with the provided data.
        
        Args:
            data (Series): A pandas Series containing disaster event information.
        """

        self.disaster_subtype = data['Disaster Subtype']
        self.country = data['Country']
        self.latitude = data['Latitude']
        self.longitude = data['Longitude']
        self.start_date = data['start_date']

        self.xyz_coords = self.lat_lon_to_xyz(self.latitude, self.longitude)
        self.one_hot_subtype = self.one_hot_encode_disaster_subtype(self.disaster_subtype)


    @staticmethod
    def lat_lon_to_xyz(lat, lon):
        """
        Converts latitude and longitude to 3D coordinates (x, y, z).
        
        Args:
            lat (float): Latitude in decimal degrees.
            lon (float): Longitude in decimal degrees.
        
        Returns:
            tuple: A tuple of 3D coordinates (x, y, z).
        """

        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        x = np.cos(lat_rad) * np.cos(lon_rad)
        y = np.cos(lat_rad) * np.sin(lon_rad)
        z = np.sin(lat_rad)

        return x, y, z


    def one_hot_encode_disaster_subtype(self, subtype):
        """
        One-hot encodes the disaster subtype.
        
        Args:
            subtype (str): The disaster subtype.
        
        Returns:
            list: A one-hot-encoded list representing the disaster subtype.
        """

        # All possible subtypes from the dataset
        subtypes = ['Ground movement', 'Tropical cyclone', 'Flash flood',
       'Riverine flood', 'Ash fall', 'Tsunami', 'Convective storm',
       'Landslide', 'Extra-tropical storm', 'Severe winter conditions',
       'Mudslide', 'Coastal flood', 'Heat wave', 'Avalanche',
       'Forest fire', 'Cold wave', 'Drought',
       'Land fire (Brush, Bush, Pasture)', 'Rockfall', 'Lava flow',
       'Pyroclastic flow']
        one_hot = [0] * len(subtypes)
        index = subtypes.index(subtype)
        one_hot[index] = 1

        return one_hot


    def get_time_range(self, days_before, days_after):
        """
        Gets the time range before and after the disaster event.
        
        Args:
            days_before (int): The number of days before the disaster event.
            days_after (int): The number of days after the disaster
            
        Returns:
            tuple: A tuple containing two tuples, each representing the start and end dates of the time range.
        """

        time_range_A_start = self.start_date - timedelta(days=days_before)
        time_range_A_end = self.start_date - timedelta(days=1)
        time_range_B_start = self.start_date
        time_range_B_end = self.start_date + timedelta(days=days_after)

        return (time_range_A_start, time_range_A_end), (time_range_B_start, time_range_B_end)


    def haversine_distance(self, lat2, lon2):
        """
        Calculates the Haversine distance between the disaster location and a given latitude and longitude.

        Args:
            lat2 (float): Latitude of the second point.
            lon2 (float): Longitude of the second point.

        Returns:
            float: The Haversine distance between the disaster location and the given point in kilometers.
        """

        r = 6371 # radius of the earth

        phi1 = np.radians(self.latitude)
        phi2 = np.radians(lat2)
        delta_phi = np.radians(lat2 - self.latitude)
        delta_lambda = np.radians(lon2 - self.longitude)
        a = np.sin(delta_phi/2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda/2)**2
        res = r * (2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))

        return np.round(res, 2)


class TweetRaster:
    """
    A class to represent a raster grid of tweets in the Mollweide projection.
    
    The class reads tweet data, processes it, and organizes it into raster grid cells
    for further analysis and visualization.
    
    Attributes:
        resolution (float): The resolution of the raster grid in meters.
        mollweide_proj (CRS): The Mollweide projection used for the raster grid.
        basemap (GeoDataFrame): The basemap data in the Mollweide projection.
        raster (GeoDataFrame): The raster grid cells as a GeoDataFrame.
        tweets (GeoDataFrame): The tweet data processed and assigned to raster grid cells.
    """

    def __init__(self, resolution):
        """
        Initializes a TweetRaster object with the specified raster grid resolution.
        
        Args:
            resolution (float): The resolution of the raster grid in meters.
        """

        # Set the Mollweide projection for the raster grid
        self.mollweide_proj = CRS.from_proj4('+proj=moll +lon_0=0 +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs')

        # Read and reproject the basemap data to the Mollweide projection
        self.basemap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres')).to_crs(self.mollweide_proj)

        # Set the raster grid resolution
        self.resolution = resolution

        # Initialize raster, tweets and disaster attributes as None
        self.raster = None
        self.tweets = None
        self.tweets_unfiltered = None
        self.disasters = None
        self.training_data = dict()

        # Build the raster grid
        self.build_raster()


    def build_raster(self):
        """
        Builds the raster grid for the map using the specified resolution.
        
        The grid is built using shapely polygons and the Mollweide projection.
        The raster grid is then stored as a GeoDataFrame in the `raster` attribute.
        """

        # Define the bounding box for the Mollweide projection
        xmin, ymin, xmax, ymax = (-18048997.159878, -9009968.760470, 18048997.159878, 9009968.760470)

        # Calculate the number of pixels in the x and y directions based on the resolution
        num_pixels_x = int(np.ceil((xmax - xmin) / self.resolution))
        num_pixels_y = int(np.ceil((ymax - ymin) / self.resolution))

        # Create a list of shapely polygons representing the grid cells in the raster
        polygons = [
            Polygon([(x, y), (x + self.resolution, y), (x + self.resolution, y + self.resolution), (x, y + self.resolution)])
            for x in np.arange(xmin, xmax, self.resolution)
            for y in np.arange(ymin, ymax, self.resolution)
        ]

        # Create a GeoDataFrame containing the grid cell polygons and their corresponding CRS
        raster = gpd.GeoDataFrame({'geometry': polygons}, crs=self.mollweide_proj)

        # Compute the centroid for each polygon
        raster['pixel_centroid'] = raster['geometry'].centroid

        # Convert the centroids to crs=4326
        raster['pixel_centroid'] = raster['pixel_centroid'].to_crs(epsg=4326)

        raster = raster.reset_index().rename(columns={'index': 'pixel_id'})

        # Store the raster grid in the class attribute
        self.raster = raster


    def load_tweets(self, path, longitude_column, latitude_column, crs, filter_before='1900-01-01'):
        """
        Loads tweet data from a CSV file, processes it, and assigns tweets to raster grid cells.
        
        Args:
            path (str): Path to the CSV file containing the tweet data.
            longitude_column (str): Name of the column containing longitude values.
            latitude_column (str): Name of the column containing latitude values.
            crs (str or dict): Coordinate reference system of the input data.
        """

        # Read tweet data from the CSV file and drop rows with missing longitude or latitude values
        tweets = pd.read_csv(path)
        tweets = tweets.dropna(subset=[longitude_column, latitude_column])

        # Create shapely Point geometries from longitude and latitude columns
        geometry = gpd.points_from_xy(tweets[longitude_column], tweets[latitude_column])

        # Convert the DataFrame to a GeoDataFrame with the specified CRS
        tweets = gpd.GeoDataFrame(tweets, geometry=geometry, crs=crs)

        # Reproject the tweet data to the Mollweide projection
        tweets = tweets.to_crs(self.mollweide_proj)

        # Convert 'created_at' column to datetime
        tweets['created_at'] = pd.to_datetime(tweets['created_at'])
        tweets['created_at'] = tweets['created_at'].dt.tz_convert(None)

        # Filter out tweets before a specified date
        tweets = tweets.loc[tweets['created_at'] >= pd.to_datetime(filter_before)]

        # Map the 'aggressiveness' and 'stance' columns to numerical values
        tweets['aggressiveness'] = tweets['aggressiveness'].map({'not aggressive': -1, 'aggressive': 1})
        tweets['stance'] = tweets['stance'].map({'denier': -1, 'neutral': 0, 'believer': 1})

        # Perform a spatial join between the raster grid and the tweet data, assigning tweets to grid cells
        tweets = gpd.sjoin(self.raster, tweets, predicate='contains')

        # Store the processed tweet data in the class attribute
        self.tweets = tweets


    def load_disasters(self, path, longitude_column, latitude_column, filter_before='1900-01-01', filter_after='2100-12-31'):
        """
        Loads disaster data from a CSV file, create Disaster objects and store them as a list of disaster objects.
        
        Args:
            path (str): Path to the CSV file containing the disaster data.
            longitude_column (str): Name of the column containing longitude values.
            latitude_column (str): Name of the column containing latitude values.
        """

        # Read disaster data from the CSV file and drop rows with missing values
        disasters = pd.read_csv(path)
        disasters = disasters.dropna(subset=[longitude_column, latitude_column, "Disaster Subtype"])

        # Convert 'start_date' column to datetime
        disasters['start_date'] = pd.to_datetime(disasters['start_date'])

        # Filter out disasters before and after e specified date
        disasters = disasters.loc[disasters['start_date'] >= pd.to_datetime(filter_before)]
        disasters = disasters.loc[disasters['start_date'] <= pd.to_datetime(filter_after)]

        # Create disaster instances and store them in a list
        disasters = [Disaster(row) for _, row in disasters.iterrows()]

        self.disasters = disasters


    def geocode_disaster_dataset(self, path, longitude_column, latitude_column):
        """
        Loads disaster data from a CSV file and geocodes disasters that have missing coordinate values.

        Args:
            path (str): Path to the CSV file containing the disaster data.
            longitude_column (str): Name of the column containing longitude values.
            latitude_column (str): Name of the column containing latitude values.
        """

        disasters = pd.read_csv(path)

        geolocator = Nominatim(user_agent="disaster_loader")

        def geocode_row(row):
            if pd.isna(row[longitude_column]) or pd.isna(row[latitude_column]):
                lat, lon = self.geocode_location(geolocator, row["Country"])
                if lat and lon:
                    row[latitude_column] = lat
                    row[longitude_column] = lon
            return row

        # Apply the geocoding function to each row of the DataFrame
        disasters = disasters.apply(geocode_row, axis=1)

        # Save the DataFrame to a new CSV file with the geocoded coordinates
        disasters.to_csv(path[:-4] + "_geocoded.csv", index=False)

        print(f"File created: {path[:-4]}_geocoded.csv)")


    def geocode_location(self, geolocator, country):
        """
        Geocodes the given country.

        Args:
            geolocator (geopy.geocoders.Nominatim): A geolocator object for geocoding.
            country (str): The country name.

        Returns:
            tuple: A tuple of (latitude, longitude).
        """

        try:
            geocode_result = geolocator.geocode(country, timeout=10)

            if geocode_result:
                return geocode_result.latitude, geocode_result.longitude
            else:
                return None, None

        except (GeocoderTimedOut, GeocoderQuotaExceeded):
            return None, None


    def select_most_active_pixels(self, n):
        """
        Selects the top n most active raster grid cells based on the number of tweets.
        
        The method filters the `tweets` attribute to keep only the tweets that
        belong to the top n most active grid cells.
        
        Args:
            n (int): The number of most active raster grid cells to select.
        """

        # Calculate the total number of tweets per raster grid cell
        total_tweets_per_pixel = self.tweets.groupby('pixel_id').size()

        # Identify the top n most active grid cells
        top_n_pixels = total_tweets_per_pixel.nlargest(n).index

        # Filter the tweet data to keep only the tweets belonging to the top n most active grid cells
        self.tweets_unfiltered = self.tweets
        self.tweets = self.tweets[self.tweets['pixel_id'].isin(top_n_pixels)]


    def reset_most_active_pixels(self):
        """
        Resets the `tweets` attribute to its original state before filtering for most active raster grid cells.

        """

        self.tweets = self.tweets_unfiltered


    def compute_change(self, time_range_A, time_range_B):
        """
        Computes the significance and type of change in tweet attributes between two time ranges for each raster grid cell.
        
         -1  meaning significant negative change
          1  meaning significant positive change
        NaN  meaning no significant change

        Args:
            time_range_A (tuple): A tuple of two datetime objects representing the start and end dates of time range A.
            time_range_B (tuple): A tuple of two datetime objects representing the start and end dates of time range B.
        
        Returns:
            GeoDataFrame: A GeoDataFrame containing the change in tweet attributes (aggressiveness, 
                          sentiment, and stance) between the two time ranges for each raster grid cell.
        """

        time_range_A_start, time_range_A_end = time_range_A
        time_range_B_start, time_range_B_end = time_range_B

        # Filter the tweet data for each time range
        tweets_A = self.tweets[(self.tweets['created_at'] >= time_range_A_start) & (self.tweets['created_at'] <= time_range_A_end)]
        tweets_B = self.tweets[(self.tweets['created_at'] >= time_range_B_start) & (self.tweets['created_at'] <= time_range_B_end)]

        # Create a DataFrame containing all unique pixel_ids
        all_pixel_ids = pd.DataFrame({'pixel_id': self.tweets['pixel_id'].unique()})
    
        # Initialize the columns for changes
        change = all_pixel_ids.copy()
        change['aggressiveness_change'] = np.nan
        change['sentiment_change'] = np.nan
        change['stance_change'] = np.nan
    
        for pixel_id in all_pixel_ids['pixel_id']:
            tweets_A_pixel = tweets_A[tweets_A['pixel_id'] == pixel_id]
            tweets_B_pixel = tweets_B[tweets_B['pixel_id'] == pixel_id]
    
            # Perform t-test for sentiment if there are tweets in both time ranges
            if not tweets_A_pixel['sentiment'].empty and not tweets_B_pixel['sentiment'].empty:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')  # Treat warnings as exceptions
                    try:
                        t_stat, p_val = ttest_ind(tweets_A_pixel['sentiment'], tweets_B_pixel['sentiment'])
                        if p_val < 0.05:  # if change is significant
                            mean_diff = tweets_B_pixel['sentiment'].mean() - tweets_A_pixel['sentiment'].mean()
                            change.loc[change['pixel_id'] == pixel_id, 'sentiment_change'] = 1 if mean_diff > 0 else -1
                    except Warning:
                        pass  # Ignore precision loss warnings
    
            # Perform Chi-Square test for homogeneity for aggressiveness if there are tweets in both time ranges
            if not tweets_A_pixel['aggressiveness'].empty and not tweets_B_pixel['aggressiveness'].empty:
                # Get the value counts for each group
                counts_A = tweets_A_pixel['aggressiveness'].value_counts()
                counts_B = tweets_B_pixel['aggressiveness'].value_counts()
                
                # Create a DataFrame from the counts and fill NaN values with 0
                data = pd.DataFrame({'A': counts_A, 'B': counts_B}).fillna(0)
                
                # Perform the Chi-Square test for homogeneity
                chi2, p_val, _, _ = chi2_contingency(data)
            
                if p_val < 0.05:  # if change is significant
                    mean_diff = tweets_B_pixel['aggressiveness'].mean() - tweets_A_pixel['aggressiveness'].mean()
                    change.loc[change['pixel_id'] == pixel_id, 'aggressiveness_change'] = 1 if mean_diff > 0 else -1
    
            # Perform Chi-Square test for homogeneity for stance if there are tweets in both time ranges
            if not tweets_A_pixel['stance'].empty and not tweets_B_pixel['stance'].empty:
                # Get the value counts for each group
                counts_A = tweets_A_pixel['stance'].value_counts()
                counts_B = tweets_B_pixel['stance'].value_counts()
                
                # Create a DataFrame from the counts and fill NaN values with 0
                data = pd.DataFrame({'A': counts_A, 'B': counts_B}).fillna(0)
                
                # Perform the Chi-Square test for homogeneity
                chi2, p_val, _, _ = chi2_contingency(data)
            
                if p_val < 0.05:  # if change is significant
                    mean_diff = tweets_B_pixel['stance'].mean() - tweets_A_pixel['stance'].mean()
                    change.loc[change['pixel_id'] == pixel_id, 'stance_change'] = 1 if mean_diff > 0 else -1
    
        # Create a new GeoDataFrame containing the changes in tweet attributes for each raster grid cell
        change = self.raster.merge(change, on='pixel_id')
    
        return change


    def create_training_data(self, attribute, days_before, days_after):
        """
        Generates the training data for a specific attribute for a machine learning task based on given disaster events.
        This function computes change vectors for a given time range and stores them along with other
        disaster features in a pandas DataFrame.

        Args:
            attribute (str): The attribute for which the training data is to be computed.
            days_before (int): The number of days before a disaster to consider for creating the time range.
            days_after (int): The number of days after a disaster to consider for creating the time range.
        """

        encoder = OneHotEncoder(sparse_output=False, categories=[[-1, 0, 1]])

        if (days_before, days_after) not in self.training_data:
            self.training_data[(days_before, days_after)] = {attribute: pd.DataFrame()}

        n_disasters = len(self.disasters)

        for disaster in self.disasters:

            print(f"Creating training data for Disaster {self.disasters.index(disaster)}/{n_disasters}.")

            time_range_A, time_range_B = disaster.get_time_range(days_before=days_before, days_after=days_after)
            changes = self.compute_change(time_range_A, time_range_B)

            data_rows = []
            for idx, change in changes.iterrows():
                # Get the disaster coordinates and one_hot_subtype
                disaster_features = np.concatenate((disaster.xyz_coords, disaster.one_hot_subtype))

                # Get the pixel coordinates
                pixel = change['pixel_centroid']
                pixel_coords = Disaster.lat_lon_to_xyz(pixel.y, pixel.x)

                # Compute the distance between the disaster and the pixel
                distance = disaster.haversine_distance(pixel.y, pixel.x)

                # Get the change for the current attribute
                attribute_change = change[f'{attribute}_change']
                if np.isnan(attribute_change):
                    attribute_change = 0
                attribute_change_encoded = encoder.fit_transform([[attribute_change]])

                # Create a data row for the current disaster-pixel pair
                data_row = np.concatenate((disaster_features, pixel_coords, [distance], attribute_change_encoded[0]))
                data_rows.append(data_row)

            df = pd.DataFrame(data_rows)
            self.training_data[(days_before, days_after)][attribute] = pd.concat([self.training_data[(days_before, days_after)][attribute], df])

        # Number of one hot encoded features
        n_features = len(self.disasters[0].one_hot_subtype)

        # Create column names
        feature_cols = ['x_disaster', 'y_disaster', 'z_disaster'] + [f'subtype_{i+1}' for i in range(n_features)] + ['x_pixel', 'y_pixel', 'z_pixel', 'distance', 'negative_change', 'no_change', 'positive_change']
        
        # Get the DataFrame for the given attribute and time range
        self.training_data[(days_before, days_after)][attribute].columns = feature_cols


    def write_training_data(self, attribute, days_before, days_after):
        """
        Writes the generated training data for a specific attribute into a CSV file.

        Args:
            attribute (str): The attribute for which the training data has been computed.
            days_before (int): The number of days before a disaster that were considered for creating the time range.
            days_after (int): The number of days after a disaster that were considered for creating the time range.
        """

        # Get the DataFrame for the given attribute and time range and write it to a CSV file
        self.training_data[(days_before, days_after)][attribute].to_csv(f'training_data_{attribute}_{days_before}b_{days_after}a.csv', index=False)

    
    def load_training_data(self, attribute, days_before, days_after):
        """
        Loads the training data for a specific attribute from a CSV file into the data structure.
    
        Args:
            attribute (str): The attribute for which the training data was computed.
            days_before (int): The number of days before a disaster that were considered for creating the time range.
            days_after (int): The number of days after a disaster that were considered for creating the time range.
        """
    
        # Define the file name based on the attribute and time range
        file_name = f'training_data_{attribute}_{days_before}b_{days_after}a.csv'
    
        # Load the CSV file into a DataFrame
        training_data_df = pd.read_csv(file_name)
    
        # Update the self.training_data dictionary with the loaded DataFrame
        self.training_data[(days_before, days_after)] = {attribute : None}
        self.training_data[(days_before, days_after)][attribute] = training_data_df


    def map_counts(self, figsize=(24, 12)):
        """
        Plot a count map based on tweet data.

        Args:
            figsize (tuple, optional): A tuple specifying the size of the figure. Defaults to (24, 12).
        """

        # Group tweets by pixel_id and count them
        counts = self.tweets.groupby('pixel_id').size().rename('count')

        # Merge the counts with the raster data
        df = self.raster.merge(counts, on='pixel_id')
    
        # Create a figure and axis with the specified figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Set the background color of the axis
        ax.set_facecolor('lightgray')

        # Plot the basemap on the axis with dark gray color, black edge color, and a linewidth of 0.5
        self.basemap.plot(ax=ax, color='darkgray', edgecolor='black', linewidth=0.5)

        # Normalize the count values
        norm = LogNorm(vmin=df['count'].min(), vmax=df['count'].max())

        df.plot(column='count', cmap='plasma', norm=norm, linewidth=0.5, edgecolor='black', legend=True, ax=ax, missing_kwds={"color": "white"})

        # Display the plot
        plt.show()


    def map_means(self, attribute, figsize=(24, 12)):
        """
        Plot a mean map of a given attribute based on tweet data.

        Args:
            attribute (str): The attribute for which to calculate the mean.
            figsize (tuple, optional): A tuple specifying the size of the figure. Defaults to (24, 12).
        """

        # Group tweets by pixel_id and compute the mean of numeric data
        means = self.tweets.groupby('pixel_id').mean(numeric_only=True)

        # Merge the computed means with the raster data
        df = self.raster.merge(means, on='pixel_id')
    
        # Create a figure and axis with the specified figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Set the background color of the axis
        ax.set_facecolor('lightgray')

        # Plot the basemap on the axis with dark gray color, black edge color, and a linewidth of 0.5
        self.basemap.plot(ax=ax, color='darkgray', edgecolor='black', linewidth=0.5)

        df.plot(column=attribute, cmap='coolwarm', linewidth=0.5, edgecolor='black', legend=True, ax=ax, missing_kwds={"color": "white"})

        # Display the plot
        plt.show()


    def map_change(self, attribute, days_before, days_after, disaster, figsize=(24, 12)):
        """
        Plot a change map of a given attribute based on tweet data for specified days before and after a disaster.

        Args:
            attribute (str): The attribute for which to calculate the change.
            days_before (int): Number of days before the disaster to consider.
            days_after (int): Number of days after the disaster to consider.
            disaster (Disaster object): The disaster event to consider.
            figsize (tuple, optional): A tuple specifying the size of the figure. Defaults to (24, 12).
        """

        # Compute time ranges before and after the disaster
        time_range_A, time_range_B = disaster.get_time_range(days_before=days_before, days_after=days_after)

        # Compute the change between the two time ranges
        change = self.compute_change(time_range_A, time_range_B)

        # Create a figure and axis with the specified figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Set the background color of the axis
        ax.set_facecolor('lightgray')

        # Plot the basemap on the axis with dark gray color, black edge color, and a linewidth of 0.5
        self.basemap.plot(ax=ax, color='darkgray', edgecolor='black', linewidth=0.5)

        change.plot(column=attribute, cmap='PiYG', linewidth=0.5, edgecolor='black', legend=True, ax=ax, missing_kwds={"color": "white"})

        # Display the plot
        plt.show()
