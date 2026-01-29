"""Image Generation API - OpenAI Compatible Image Generation Interface"""

import time
import re
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException

from app.core.auth import auth_manager
from app.core.logger import logger
from app.core.exception import GrokApiException
from app.models.openai_schema import (
    ImageGenerationRequest, 
    ImageGenerationResponse, 
    ImageData
)
from app.services.grok.client import GrokClient


router = APIRouter(prefix="/images", tags=["Image Generation"])


@router.post("/generations", response_model=ImageGenerationResponse)
async def create_image(
    request: ImageGenerationRequest,
    _: Optional[str] = Depends(auth_manager.verify)
):
    """Create image (OpenAI compatible)
    
    Uses Grok's grok-imagine-0.9 model for image generation.
    The prompt is sent to the Grok chat API which returns generated images.
    
    Args:
        request: Image generation request
        
    Returns:
        ImageGenerationResponse: List of generated images
    """
    try:
        logger.info(f"[ImageGen] Received image generation request: prompt={request.prompt[:50]}...")
        
        # Build messages for Grok client (simulate chat request)
        messages = [
            {"role": "user", "content": f"Generate an image: {request.prompt}"}
        ]
        
        # Create request for GrokClient
        grok_request = {
            "model": "grok-imagine-0.9",
            "messages": messages,
            "stream": False
        }
        
        # Call Grok API
        result = await GrokClient.openai_to_grok(grok_request)
        
        # Extract images from response
        images = _extract_images_from_response(result)
        
        if not images:
            raise GrokApiException("No images generated", "NO_IMAGES")
        
        # Limit to requested number of images
        images = images[:request.n]
        
        logger.info(f"[ImageGen] Image generation successful: {len(images)} images")
        
        return ImageGenerationResponse(
            created=int(time.time()),
            data=[
                ImageData(
                    url=img_url,
                    revised_prompt=request.prompt
                )
                for img_url in images
            ]
        )
        
    except GrokApiException as e:
        logger.error(f"[ImageGen] Grok API error: {e.message}")
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
        logger.error(f"[ImageGen] Unexpected error: {str(e)}")
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


def _extract_images_from_response(response) -> list:
    """Extract image URLs from Grok response
    
    Grok returns images in various formats:
    - In the message content as markdown: ![image](url)
    - In structured response with image_urls
    """
    images = []
    
    # Handle Pydantic models by converting to dict or accessing attributes
    if hasattr(response, 'choices'):
        # Pydantic model
        choices = response.choices or []
        for choice in choices:
            message = choice.message if hasattr(choice, 'message') else None
            if message:
                content = message.content if hasattr(message, 'content') else ""
                if content:
                    # Extract markdown image URLs
                    md_images = re.findall(r'!\[.*?\]\((https?://[^\s\)]+)\)', content)
                    images.extend(md_images)
                    
                    # Extract plain URLs that look like images
                    url_images = re.findall(r'(https?://[^\s\)]+\.(?:jpg|jpeg|png|gif|webp))', content, re.IGNORECASE)
                    images.extend(url_images)
                    
                    # Check for grok.com image URLs
                    grok_images = re.findall(r'(https?://(?:assets\.)?grok\.com/[^\s\)\"]+)', content)
                    images.extend(grok_images)
    elif isinstance(response, dict):
        # Check for image URLs in choices
        choices = response.get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            
            # Extract markdown image URLs
            md_images = re.findall(r'!\[.*?\]\((https?://[^\s\)]+)\)', content)
            images.extend(md_images)
            
            # Extract plain URLs that look like images
            url_images = re.findall(r'(https?://[^\s\)]+\.(?:jpg|jpeg|png|gif|webp))', content, re.IGNORECASE)
            images.extend(url_images)
            
            # Check for grok.com image URLs
            grok_images = re.findall(r'(https?://(?:assets\.)?grok\.com/[^\s\)\"]+)', content)
            images.extend(grok_images)
    
    # Deduplicate while preserving order
    seen = set()
    unique_images = []
    for img in images:
        if img not in seen:
            seen.add(img)
            unique_images.append(img)
    
    return unique_images
