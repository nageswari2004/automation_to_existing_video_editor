import os
import requests
from dotenv import load_dotenv
from typing import Optional, Dict, List, Union
import json
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class VimeoAPI:
    def __init__(self, max_retries: int = 5, retry_delay: int = 5):
        load_dotenv()
        self.access_token = os.getenv('VIMEO_ACCESS_TOKEN')
        if not self.access_token:
            raise ValueError("VIMEO_ACCESS_TOKEN not found in environment variables")
            
        self.base_url = 'https://api.vimeo.com'
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Content-Type': 'application/json',
            'Accept': 'application/vnd.vimeo.*+json;version=3.4'
        }
        
        # Set up retry strategy with longer delays
        self.session = requests.Session()
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=retry_delay,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE", "PATCH"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _make_request(self, endpoint: str, method: str = 'GET', params: Optional[Dict] = None, data: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make a request to the Vimeo API with enhanced error handling and retry logic
        """
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(
                method=method,
                url=url,
                headers=self.headers,
                params=params,
                json=data
            )
            
            if response.status_code == 401:
                raise ValueError("Authentication failed. Please check your access token.")
            elif response.status_code == 403:
                raise ValueError("Access forbidden. Please check your token permissions.")
            elif response.status_code == 404:
                raise ValueError("Resource not found.")
            elif response.status_code == 429:
                retry_after = response.headers.get('Retry-After', '60')
                raise ValueError(f"Rate limit exceeded. Please try again after {retry_after} seconds.")
            elif response.status_code == 503:
                raise ValueError("Vimeo search service is temporarily unavailable. Please try again later.")
            elif response.status_code >= 500:
                raise ValueError(f"Vimeo server error (HTTP {response.status_code}). Please try again later.")
                
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {str(e)}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_data = e.response.json()
                    if 'developer_message' in error_data:
                        print(f"Error details: {error_data['developer_message']}")
                except:
                    print(f"Error response: {e.response.text}")
            return None

    def search_videos(self, query: str, per_page: int = 10, page: int = 1, 
                     filter_type: str = 'CC', sort: str = 'relevant') -> Optional[Dict]:
        """
        Search for videos on Vimeo with multiple fallback strategies
        
        Args:
            query: Search query string
            per_page: Number of results per page (max 100)
            page: Page number
            filter_type: Filter type (CC, VOD, etc.)
            sort: Sort order (relevant, newest, oldest, etc.)
        """
        # Try different search strategies
        search_strategies = [
            # Strategy 1: Full search with all parameters
            {
                'params': {
                    'query': query,
                    'per_page': min(per_page, 100),
                    'page': page,
                    'filter': filter_type,
                    'sort': sort
                }
            },
            # Strategy 2: Search without sorting
            {
                'params': {
                    'query': query,
                    'per_page': min(per_page, 100),
                    'page': page,
                    'filter': filter_type
                }
            },
            # Strategy 3: Search without filter
            {
                'params': {
                    'query': query,
                    'per_page': min(per_page, 100),
                    'page': page
                }
            },
            # Strategy 4: Search with minimal parameters
            {
                'params': {
                    'query': query,
                    'per_page': min(per_page, 100)
                }
            }
        ]

        for strategy in search_strategies:
            try:
                print(f"\nTrying search strategy: {strategy['params']}")
                time.sleep(3)  # Wait between attempts
                result = self._make_request('/videos', params=strategy['params'])
                if result and 'data' in result and result['data']:
                    return result
            except Exception as e:
                print(f"Strategy failed: {str(e)}")
                continue

        print("All search strategies failed. No results found.")
        return None

    def get_video_info(self, video_id: str) -> Optional[Dict]:
        """
        Get detailed information about a specific video
        """
        return self._make_request(f'/videos/{video_id}')

    def get_user_videos(self, user_id: str, per_page: int = 10, page: int = 1) -> Optional[Dict]:
        """
        Get videos from a specific user
        """
        params = {
            'per_page': min(per_page, 100),
            'page': page
        }
        return self._make_request(f'/users/{user_id}/videos', params=params)

    def get_video_download_url(self, video_id: str) -> Optional[str]:
        """
        Get the download URL for a video (if available)
        """
        data = self.get_video_info(video_id)
        if data and 'download' in data and data['download']:
            return data['download'][0]['link']
        return None

    def upload_video(self, file_path: str, name: Optional[str] = None, 
                    description: Optional[str] = None, privacy: str = 'anybody') -> Optional[str]:
        """
        Upload a video to Vimeo with privacy settings
        
        Args:
            file_path: Path to the video file
            name: Video title
            description: Video description
            privacy: Privacy setting (anybody, password, disable, etc.)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # First, create an upload
        data = {
            'upload': {
                'approach': 'tus',
                'size': os.path.getsize(file_path)
            },
            'privacy': {
                'view': privacy
            }
        }
        if name:
            data['name'] = name
        if description:
            data['description'] = description

        response = self._make_request('/me/videos', method='POST', data=data)
        if not response:
            return None

        # Get the upload URL
        upload_url = response['upload']['upload_link']
        
        # Upload the file
        try:
            with open(file_path, 'rb') as f:
                upload_response = self.session.patch(
                    upload_url,
                    headers={
                        'Content-Type': 'application/offset+octet-stream',
                        'Tus-Resumable': '1.0.0',
                        'Upload-Offset': '0'
                    },
                    data=f
                )
                upload_response.raise_for_status()
                return response['uri'].split('/')[-1]  # Return the video ID
        except Exception as e:
            print(f"Upload failed: {str(e)}")
            return None

    def get_video_embed_code(self, video_id: str, width: int = 640, height: int = 360) -> Optional[str]:
        """
        Get the embed code for a video
        """
        data = self.get_video_info(video_id)
        if data and 'embed' in data:
            return data['embed']['html'].replace('width="640"', f'width="{width}"').replace('height="360"', f'height="{height}"')
        return None

    def get_video_thumbnails(self, video_id: str) -> Optional[List[Dict]]:
        """
        Get available thumbnails for a video
        """
        data = self.get_video_info(video_id)
        if data and 'pictures' in data and 'sizes' in data['pictures']:
            return data['pictures']['sizes']
        return None 