"""Video Generation API - Video Generation Interface"""

import time
import re
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import auth_manager
from app.core.logger import logger
from app.core.exception import GrokApiException
from app.models.openai_schema import (
    VideoGenerationRequest, 
    VideoGenerationResponse, 
    VideoData
)
from app.services.grok.client import GrokClient


router = APIRouter(prefix="/videos", tags=["Video Generation"])


@router.post("/generations", response_model=VideoGenerationResponse)
async def create_video(
    request: VideoGenerationRequest,
    _: Optional[str] = Depends(auth_manager.verify)
):
    """Create video
    
    Uses Grok's grok-imagine-0.9 model for video generation.
    Supports text-to-video and image-to-video generation.
    
    Args:
        request: Video generation request
        
    Returns:
        VideoGenerationResponse: Generated video information
    """
    try:
        logger.info(f"[VideoGen] Received video generation request: prompt={request.prompt[:50]}...")
        
        # Build messages for Grok client
        content_parts = []
        
        # If image_url is provided, include it for image-to-video
        if request.image_url:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": request.image_url}
            })
        
        content_parts.append({
            "type": "text",
            "text": f"Create a video: {request.prompt}"
        })
        
        messages = [
            {"role": "user", "content": content_parts}
        ]
        
        # Create request for GrokClient
        grok_request = {
            "model": "grok-imagine-0.9",
            "messages": messages,
            "stream": False
        }
        
        # Call Grok API
        result = await GrokClient.openai_to_grok(grok_request)
        
        # Extract videos from response
        videos = _extract_videos_from_response(result)
        
        if not videos:
            raise GrokApiException("No videos generated", "NO_VIDEOS")
        
        logger.info(f"[VideoGen] Video generation successful: {len(videos)} videos")
        
        return VideoGenerationResponse(
            created=int(time.time()),
            data=[
                VideoData(
                    url=video_url,
                    revised_prompt=request.prompt
                )
                for video_url in videos
            ]
        )
        
    except GrokApiException as e:
        logger.error(f"[VideoGen] Grok API error: {e.message}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": e.message,
                    "type": e.error_code or "grok_api_error",
                    "code": e.error_code or "unknown"
                }
            }
        )
    except Exception as e:
        logger.error(f"[VideoGen] Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "internal_error",
                    "code": "internal_error"
                }
            }
        )


def _extract_videos_from_response(response) -> list:
    """Extract video URLs from Grok response
    
    Grok returns videos in various formats:
    - In the message content as markdown or plain URLs
    - In structured response with video_urls
    """
    videos = []
    
    # Handle Pydantic models by accessing attributes
    if hasattr(response, 'choices'):
        # Pydantic model
        choices = response.choices or []
        for choice in choices:
            message = choice.message if hasattr(choice, 'message') else None
            if message:
                content = message.content if hasattr(message, 'content') else ""
                if content:
                    # Extract video URLs (common video extensions)
                    video_urls = re.findall(r'(https?://[^\s\)\"]+\.(?:mp4|webm|mov|avi))', content, re.IGNORECASE)
                    videos.extend(video_urls)
                    
                    # Extract grok.com video URLs
                    grok_videos = re.findall(r'(https?://(?:assets\.)?grok\.com/[^\s\)\"]*(?:video|\.mp4|\.webm)[^\s\)\"]*)', content, re.IGNORECASE)
                    videos.extend(grok_videos)
    elif isinstance(response, dict):
        # Check for video URLs in choices
        choices = response.get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            
            # Extract video URLs (common video extensions)
            video_urls = re.findall(r'(https?://[^\s\)\"]+\.(?:mp4|webm|mov|avi))', content, re.IGNORECASE)
            videos.extend(video_urls)
            
            # Extract grok.com video URLs
            grok_videos = re.findall(r'(https?://(?:assets\.)?grok\.com/[^\s\)\"]*(?:video|\.mp4|\.webm)[^\s\)\"]*)', content, re.IGNORECASE)
            videos.extend(grok_videos)
    
    # Deduplicate while preserving order
    seen = set()
    unique_videos = []
    for vid in videos:
        if vid not in seen:
            seen.add(vid)
            unique_videos.append(vid)
    
    return unique_videos
