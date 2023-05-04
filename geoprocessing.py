import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from pyproj import CRS

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
        self.latitude = data['Latitude']
        self.longitude = data['Longitude']
        self.start_date = datetime.strptime(data['start_date'], '%Y-%m-%d')

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
       'Riverine flood', 'Ash fall', 'Tsunami', None, 'Convective storm',
       'Landslide', 'Extra-tropical storm', 'Severe winter conditions',
       'Mudslide', 'Coastal flood', 'Heat wave', 'Avalanche',
       'Forest fire', 'Cold wave', 'Drought',
       'Land fire (Brush, Bush, Pasture)', 'Rockfall', 'Lava flow',
       'Pyroclastic flow']
        one_hot = [0] * len(subtypes)
        index = subtypes.index(subtype)
        one_hot[index] = 1
        return one_hot

    def get_feature_vector(self):
        """
        Gets the feature vector for the disaster event.
        
        Returns:
            ndarray: A numpy array containing the feature vector.
        """

        return np.concatenate((self.xyz_coords, self.one_hot_subtype))

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

        # Initialize raster and tweets attributes as None
        self.raster = None
        self.tweets = None

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
        raster = raster.reset_index().rename(columns={'index': 'pixel_id'})

        # Store the raster grid in the class attribute
        self.raster = raster


    def load_tweets(self, path, longitude_column, latitude_column, crs):
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

        # Map the 'aggressiveness' and 'stance' columns to numerical values
        tweets['aggressiveness'] = tweets['aggressiveness'].map({'not aggressive': -1, 'aggressive': 1})
        tweets['stance'] = tweets['stance'].map({'denier': -1, 'neutral': 0, 'believer': 1})

        # Perform a spatial join between the raster grid and the tweet data, assigning tweets to grid cells
        tweets = gpd.sjoin(self.raster, tweets, predicate='contains')

        # Store the processed tweet data in the class attribute
        self.tweets = tweets


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
        self.tweets = self.tweets[self.tweets['pixel_id'].isin(top_n_pixels)]


    def compute_difference(self, time_range_A, time_range_B):
        """
        Computes the difference in tweet attributes between two time ranges for each raster grid cell.
        
        Args:
            time_range_A (tuple): A tuple of two datetime objects representing the start and end dates of time range A.
            time_range_B (tuple): A tuple of two datetime objects representing the start and end dates of time range B.
        
        Returns:
            GeoDataFrame: A GeoDataFrame containing the difference in tweet attributes (aggressiveness, 
                          sentiment, and stance) between the two time ranges for each raster grid cell.
        """

        time_range_A_start, time_range_A_end = time_range_A
        time_range_B_start, time_range_B_end = time_range_B

        # Filter the tweet data for each time range
        tweets_A = self.tweets[(self.tweets['created_at'] >= time_range_A_start) & (self.tweets['created_at'] <= time_range_A_end)]
        tweets_B = self.tweets[(self.tweets['created_at'] >= time_range_B_start) & (self.tweets['created_at'] <= time_range_B_end)]

        # Group the tweet data by raster grid cell and calculate the mean aggressiveness, sentiment, and stance for each group
        grouped_A = tweets_A.groupby('pixel_id').agg({'aggressiveness': 'mean', 'sentiment': 'mean', 'stance': 'mean', 'pixel_id': 'count'}).rename(columns={'pixel_id': 'count_A'}).reset_index()
        grouped_B = tweets_B.groupby('pixel_id').agg({'aggressiveness': 'mean', 'sentiment': 'mean', 'stance': 'mean', 'pixel_id': 'count'}).rename(columns={'pixel_id': 'count_B'}).reset_index()

        # Create a DataFrame containing all unique pixel_ids
        all_pixel_ids = pd.DataFrame({'pixel_id': self.tweets['pixel_id'].unique()})

        # Merge the grouped data with the all_pixel_ids DataFrame to ensure all pixel_ids are included
        grouped_A = all_pixel_ids.merge(grouped_A, on='pixel_id', how='left')
        grouped_B = all_pixel_ids.merge(grouped_B, on='pixel_id', how='left')

        # Merge the two grouped DataFrames to compute the difference in attributes
        comparison = grouped_A.merge(grouped_B, on='pixel_id', suffixes=('_A', '_B'))

        # Calculate the difference in aggressiveness, sentiment, and stance between the two time ranges
        comparison['aggressiveness_diff'] = comparison['aggressiveness_B'] - comparison['aggressiveness_A']
        comparison['sentiment_diff'] = comparison['sentiment_B'] - comparison['sentiment_A']
        comparison['stance_diff'] = comparison['stance_B'] - comparison['stance_A']

        # Create a new GeoDataFrame containing the differences in tweet attributes for each raster grid cell
        difference = comparison[['pixel_id', 'aggressiveness_diff', 'sentiment_diff', 'stance_diff']]
        difference = self.raster.merge(difference, on='pixel_id')

        return difference


    def plot_values(self, df, column, figsize=(24, 12)):
        """
        Plots the values of a specified column in the input GeoDataFrame on a Mollweide-projected map.
        
        Args:
            df (GeoDataFrame): The GeoDataFrame containing the data to plot.
            column (str): The name of the column containing the values to plot.
            figsize (tuple, optional): A tuple of width and height in inches for the figure. Defaults to (24, 12).
        """

        # Create a figure and axis with the specified figsize
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Set the background color of the axis
        ax.set_facecolor('lightgray')

        # Plot the basemap on the axis with dark gray color, black edge color, and a linewidth of 0.5
        self.basemap.plot(ax=ax, color='darkgray', edgecolor='black', linewidth=0.5)

        # Plot the input GeoDataFrame on the axis using the specified column to define the color map
        df.plot(column=column, cmap='coolwarm', linewidth=0.5, edgecolor='black', legend=True, ax=ax, missing_kwds={"color": "white"})

        # Display the plot
        plt.show()
