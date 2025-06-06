from vimeo_client import VimeoAPI
import os
from dotenv import load_dotenv
from typing import Optional
import time

def test_vimeo_api():
    try:
        # First, verify the environment variable
        load_dotenv()
        token = os.getenv('VIMEO_ACCESS_TOKEN')
        print("\nChecking environment setup...")
        print(f"Access token found: {'Yes' if token else 'No'}")
        if token:
            print(f"Token length: {len(token)}")
            print(f"Token starts with: {token[:10]}...")
        
        # Initialize the API with increased retry settings
        print("\nInitializing Vimeo API...")
        vimeo = VimeoAPI(max_retries=5, retry_delay=5)
        
        # Test different search terms
        search_terms = [
            "nature",
            "documentary",
            "travel",
            "music",
            "art"
        ]
        
        for term in search_terms:
            print(f"\n{'='*50}")
            print(f"Testing search for term: '{term}'")
            print(f"{'='*50}")
            
            try:
                results = vimeo.search_videos(term, per_page=5)
                if results and 'data' in results:
                    print(f"\nFound {len(results['data'])} videos for '{term}'")
                    for video in results['data'][:2]:  # Show first 2 results
                        print(f"\nTitle: {video.get('name')}")
                        print(f"Link: {video.get('link')}")
                        print(f"Duration: {video.get('duration')} seconds")
                        if 'created_time' in video:
                            print(f"Created: {video.get('created_time')}")
                else:
                    print(f"No results found for '{term}'")
                
                # Add a delay between searches
                if term != search_terms[-1]:  # Don't wait after the last term
                    print("\nWaiting 5 seconds before next search...")
                    time.sleep(5)
                    
            except Exception as e:
                print(f"Error searching for '{term}': {str(e)}")
                continue
        
        # If we found any videos, get details for the first one
        if results and 'data' in results and results['data']:
            first_video = results['data'][0]
            video_id = first_video['uri'].split('/')[-1]
            print(f"\n{'='*50}")
            print(f"Getting details for video {video_id}")
            print(f"{'='*50}")
            
            try:
                video_info = vimeo.get_video_info(video_id)
                if video_info:
                    print(f"\nTitle: {video_info.get('name')}")
                    print(f"Description: {video_info.get('description')}")
                    print(f"Duration: {video_info.get('duration')} seconds")
                    print(f"Views: {video_info.get('stats', {}).get('plays', 0)}")
                    
                    # Get embed code
                    print("\nGetting embed code:")
                    embed_code = vimeo.get_video_embed_code(video_id, width=800, height=450)
                    if embed_code:
                        print("Embed code generated successfully")
                        print(f"Code length: {len(embed_code)} characters")
                    
                    # Get thumbnails
                    print("\nGetting available thumbnails:")
                    thumbnails = vimeo.get_video_thumbnails(video_id)
                    if thumbnails:
                        print(f"Found {len(thumbnails)} thumbnail sizes")
                        for thumb in thumbnails[:2]:  # Show first 2 thumbnails
                            print(f"Size: {thumb.get('width')}x{thumb.get('height')}")
                            print(f"Link: {thumb.get('link')}")
            except Exception as e:
                print(f"Error getting video details: {str(e)}")
            
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print("\nFull error traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_vimeo_api() 