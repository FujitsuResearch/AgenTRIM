# tools/dalle_tool.py
import os
import requests
import json
import time
from typing import Dict, List, Optional, Any, Union, ClassVar
from PIL import Image
from io import BytesIO
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class DallEImageInput(BaseModel):
    """Inputs for DALL-E image generation."""

    prompt: str = Field(..., description="Prompt to generate the image from.")
    size: str = Field(
        default="1024x1024",
        description="Size of the image. Options: 1024x1024, 512x512, 256x256.",
    )


class DallETool(BaseTool):
    """Tool for generating images using DALL-E via Azure OpenAI."""

    # Properly annotate inherited fields with type annotations
    name: ClassVar[str] = "dalle_image_generator"
    description: ClassVar[str] = (
        "Generate images using DALL-E. Provide a detailed prompt describing the image you want to create."
    )
    args_schema: ClassVar[type[BaseModel]] = DallEImageInput

    # Add fields for API credentials with proper annotations
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    api_version: Optional[str] = None
    deployment_name: Optional[str] = None

    def __init__(
        self, api_key=None, api_base=None, api_version=None, deployment_name=None
    ):
        """
        Initialize the DALL-E tool with custom Azure parameters.

        Args:
            api_key (str, optional): Azure API key for DALL-E. Defaults to environment variable.
            api_base (str, optional): Azure API base URL for DALL-E. Defaults to environment variable.
            api_version (str, optional): Azure API version for DALL-E. Defaults to environment variable.
            deployment_name (str, optional): DALL-E deployment name. Defaults to environment variable.
        """
        # Initialize parent class first
        super().__init__()

        # Set API parameters
        self.api_key = api_key or os.environ.get("DALLE_API_KEY")
        self.api_base = api_base or os.environ.get("DALLE_API_BASE")
        self.api_version = api_version or os.environ.get("DALLE_API_VERSION")
        self.deployment_name = deployment_name or os.environ.get(
            "DALLE_DEPLOYMENT_NAME"
        )

    def _run(self, prompt: str, size: str = "1024x1024") -> str:
        """
        Generate an image using DALL-E and return the local file path.

        Args:
            prompt (str): Description of the image to generate
            size (str): Size of the image, default is 1024x1024

        Returns:
            str: Path to the saved image or error message
        """
        try:
            # Create a directory to store the images if it doesn't exist
            save_dir = os.path.join(os.getcwd(), "dalle_images")
            os.makedirs(save_dir, exist_ok=True)

            # Verify API credentials
            if not all([self.api_key, self.api_base, self.api_version]):
                return "Error: Azure OpenAI API credentials for DALL-E not found. Please set DALLE_API_KEY, DALLE_API_BASE, and DALLE_API_VERSION environment variables."

            # Use the correct endpoint format with deployment name
            if self.deployment_name:
                # Format from your provided URL:
                # https://genimagedetectionopenai.openai.azure.com/openai/deployments/dall-e-3/images/generations?api-version=2024-02-01
                endpoint = f"{self.api_base}/openai/deployments/{self.deployment_name}/images/generations?api-version={self.api_version}"
            else:
                # Fallback to generic endpoint format
                endpoint = f"{self.api_base}/openai/images/generations?api-version={self.api_version}"

            print(f"Using DALL-E endpoint: {endpoint}")

            # Request headers
            headers = {"Content-Type": "application/json", "api-key": self.api_key}

            # Request body
            data = {"prompt": prompt, "n": 1, "size": size}

            # Make the API call
            print(f"Sending request to DALL-E API with prompt: {prompt[:50]}...")
            response = requests.post(endpoint, headers=headers, json=data)

            # Print response status for debugging
            print(f"DALL-E API response status: {response.status_code}")

            if response.status_code == 200:  # Success (immediate response)
                response_data = response.json()
                # Handle direct response format
                if "data" in response_data and len(response_data["data"]) > 0:
                    image_url = response_data["data"][0].get("url")

                    if image_url:
                        # Download the image
                        image_response = requests.get(image_url)
                        image = Image.open(BytesIO(image_response.content))

                        # Generate a timestamped filename
                        timestamp = int(time.time())
                        image_name = f"dalle_image_{timestamp}.png"
                        image_path = os.path.join(save_dir, image_name)

                        # Save the image
                        image.save(image_path)

                        return image_path

            elif response.status_code == 202:  # Accepted (asynchronous operation)
                # Get the operation ID
                operation_id = response.headers.get("Operation-Location", "").split(
                    "/"
                )[-1]

                # Poll for the result
                status_endpoint = f"{self.api_base}/openai/operations/images/{operation_id}?api-version={self.api_version}"

                print(f"Polling operation at: {status_endpoint}")

                max_retries = 60  # Maximum number of retries
                retry_delay = 1  # Initial delay in seconds

                for _ in range(max_retries):
                    time.sleep(retry_delay)
                    status_response = requests.get(
                        status_endpoint, headers={"api-key": self.api_key}
                    )
                    status_data = status_response.json()

                    # Check if the operation is successful
                    if status_data.get("status") == "succeeded":
                        # Get the image URL
                        image_url = (
                            status_data.get("result", {})
                            .get("data", [{}])[0]
                            .get("url")
                        )

                        if image_url:
                            # Download the image
                            image_response = requests.get(image_url)
                            image = Image.open(BytesIO(image_response.content))

                            # Generate a timestamped filename
                            timestamp = int(time.time())
                            image_name = f"dalle_image_{timestamp}.png"
                            image_path = os.path.join(save_dir, image_name)

                            # Save the image
                            image.save(image_path)

                            return image_path

                    # Check if the operation failed
                    if status_data.get("status") == "failed":
                        return f"Image generation failed: {status_data.get('error', {}).get('message')}"

                    # Increase the delay with exponential backoff (capped at 10 seconds)
                    retry_delay = min(retry_delay * 1.5, 10)

                return "Image generation timed out"
            else:
                # Print detailed error information for debugging
                error_message = f"API request failed: {response.status_code}"
                try:
                    error_details = response.json()
                    error_message += f" - {json.dumps(error_details)}"
                except:
                    error_message += f" - {response.text}"

                print(f"DALL-E API error: {error_message}")
                return error_message

        except Exception as e:
            print(f"Exception in DALL-E image generation: {str(e)}")
            return f"Error generating image: {str(e)}"

    def _arun(self, prompt: str, size: str = "1024x1024") -> str:
        """Async implementation would go here, but we're using the sync version."""
        raise NotImplementedError("Async version not implemented")
