from flask import Flask
import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

def handler(request):
    """Handle incoming requests in a serverless environment"""
    try:
        with app.test_client() as test_client:
            # Get the path and method from the request
            path = request.get('path', '/')
            method = request.get('method', 'GET')
            headers = request.get('headers', {})
            body = request.get('body', '')
            
            # Handle multipart/form-data
            if headers.get('content-type', '').startswith('multipart/form-data'):
                files = request.get('files', {})
                form = request.get('form', {})
                
                response = test_client.open(
                    path=path,
                    method=method,
                    headers={k: v for k, v in headers.items() if k.lower() != 'content-length'},
                    data=form,
                    files=files
                )
            else:
                response = test_client.open(
                    path=path,
                    method=method,
                    headers=headers,
                    data=body
                )
            
            return {
                'statusCode': response.status_code,
                'headers': dict(response.headers),
                'body': response.get_data(as_text=True)
            }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': str(e),
            'headers': {
                'Content-Type': 'text/plain',
            }
        } 