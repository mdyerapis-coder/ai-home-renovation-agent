import os
import logging
from google import genai
from google.genai import types
from google.adk.tools import ToolContext
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


def get_next_version_number(tool_context: ToolContext, asset_name: str) -> int:
    """Get the next version number for a given asset name."""
    asset_versions = tool_context.state.get("asset_versions", {})
    current_version = asset_versions.get(asset_name, 0)
    next_version = current_version + 1
    return next_version


def update_asset_version(tool_context: ToolContext, asset_name: str, version: int, filename: str) -> None:
    """Update the version tracking for an asset."""
    if "asset_versions" not in tool_context.state:
        tool_context.state["asset_versions"] = {}
    if "asset_filenames" not in tool_context.state:
        tool_context.state["asset_filenames"] = {}
    
    tool_context.state["asset_versions"][asset_name] = version
    tool_context.state["asset_filenames"][asset_name] = filename
    
    asset_history_key = f"{asset_name}_history"
    if asset_history_key not in tool_context.state:
        tool_context.state[asset_history_key] = []
    tool_context.state[asset_history_key].append({"version": version, "filename": filename})


def create_versioned_filename(asset_name: str, version: int, file_extension: str = "png") -> str:
    """Create a versioned filename for an asset."""
    return f"{asset_name}_v{version}.{file_extension}"


def get_asset_versions_info(tool_context: ToolContext) -> str:
    """Get information about all asset versions in the session."""
    asset_versions = tool_context.state.get("asset_versions", {})
    if not asset_versions:
        return "No renovation renderings have been created yet."
    
    info_lines = ["Current renovation renderings:"]
    for asset_name, current_version in asset_versions.items():
        history_key = f"{asset_name}_history"
        history = tool_context.state.get(history_key, [])
        total_versions = len(history)
        latest_filename = tool_context.state.get("asset_filenames", {}).get(asset_name, "Unknown")
        info_lines.append(f"  • {asset_name}: {total_versions} version(s), latest is v{current_version} ({latest_filename})")
    
    return "\n".join(info_lines)


async def load_reference_image(tool_context: ToolContext, filename: str):
    """Load a reference image artifact by filename."""
    try:
        loaded_part = await tool_context.load_artifact(filename)
        if loaded_part:
            logger.info(f"Successfully loaded reference image: {filename}")
            return loaded_part
        else:
            logger.warning(f"Reference image not found: {filename}")
            return None
    except Exception as e:
        logger.error(f"Error loading reference image {filename}: {e}")
        return None


class GenerateRenovationRenderingInput(BaseModel):
    prompt: str = Field(..., description="A detailed description of the renovated space to generate. Include room type, style, colors, materials, fixtures, lighting, and layout.")
    aspect_ratio: str = Field(default="16:9", description="The desired aspect ratio, e.g., '1:1', '16:9', '9:16'. Default is 16:9 for room photos.")
    asset_name: str = Field(default="renovation_rendering", description="Base name for the rendering (will be versioned automatically). Use descriptive names like 'kitchen_modern_farmhouse' or 'bathroom_spa'.")
    current_room_photo: str = Field(default=None, description="Optional: filename of the current room photo to use as reference for layout/structure.")
    inspiration_image: str = Field(default=None, description="Optional: filename of an inspiration image to guide the style.")


class EditRenovationRenderingInput(BaseModel):
    artifact_filename: str = Field(default=None, description="The filename of the rendering artifact to edit. If not provided, uses the last generated rendering.")
    prompt: str = Field(..., description="The prompt describing the desired changes (e.g., 'make cabinets darker', 'add pendant lights', 'change floor to hardwood').")
    asset_name: str = Field(default=None, description="Optional: specify asset name for the new version (defaults to incrementing current asset).")
    reference_image_filename: str = Field(default=None, description="Optional: filename of a reference image to guide the edit.")


async def _stream_and_save_rendering(
    client,
    contents,
    tool_context: ToolContext,
    asset_name: str,
    artifact_filename: str,
    version: int,
    *,
    success: str,
    save_error: str,
    empty: str,
) -> str:
    config = types.GenerateContentConfig(response_modalities=["IMAGE", "TEXT"])
    for chunk in client.models.generate_content_stream(
        model="gemini-3-pro-image-preview",
        contents=contents,
        config=config,
    ):
        if (
            chunk.candidates is None
            or chunk.candidates[0].content is None
            or chunk.candidates[0].content.parts is None
        ):
            continue

        part = chunk.candidates[0].content.parts[0]
        if part.inline_data and part.inline_data.data:
            image_part = types.Part(inline_data=part.inline_data)
            try:
                # Do NOT rebind `version` from save_artifact: that is ADK's
                # per-filename artifact revision, not our asset version counter.
                await tool_context.save_artifact(
                    filename=artifact_filename,
                    artifact=image_part,
                )
                update_asset_version(tool_context, asset_name, version, artifact_filename)
                tool_context.state["last_generated_rendering"] = artifact_filename
                tool_context.state["current_asset_name"] = asset_name
                logger.info(f"Saved rendering as artifact '{artifact_filename}' (version {version})")
                return success
            except Exception as e:
                logger.error(f"Error saving artifact: {e}")
                return f"{save_error}: {e}"
        else:
            if hasattr(chunk, "text") and chunk.text:
                logger.info(f"Model response: {chunk.text}")

    return empty


async def generate_renovation_rendering(tool_context: ToolContext, inputs: GenerateRenovationRenderingInput) -> str:
    """
    Generates a photorealistic rendering of a renovated space based on the design plan.
    
    This tool uses Gemini 3 Pro's image generation capabilities to create visual 
    representations of renovation plans. It can optionally use current room photos 
    and inspiration images as references.
    """
    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.")

    logger.info("Starting renovation rendering generation")
    try:
        client = genai.Client()
        
        # ADK may pass a dict instead of the Pydantic model.
        if isinstance(inputs, dict):
            inputs = GenerateRenovationRenderingInput(**inputs)
        
        reference_images = []
        
        if inputs.current_room_photo:
            current_photo_part = await load_reference_image(tool_context, inputs.current_room_photo)
            if current_photo_part:
                reference_images.append(current_photo_part)
                logger.info(f"Using current room photo: {inputs.current_room_photo}")
        
        if inputs.inspiration_image:
            inspiration_part = await load_reference_image(tool_context, inputs.inspiration_image)
            if inspiration_part:
                reference_images.append(inspiration_part)
                logger.info(f"Using inspiration image: {inputs.inspiration_image}")
        
        # Sole SLC rewrite (Subject, Lighting, Camera). Callers pass a content
        # description of the renovated space; do not also SLC-format in the agent.
        base_rewrite_prompt = f"""
        Create an ultra-detailed, photorealistic prompt for generating a professional interior design photograph.
        
        Original description: {inputs.prompt}
        
        **CRITICAL REQUIREMENT - PRESERVE EXACT LAYOUT:**
        The generated image MUST maintain the EXACT same room layout, structure, and spatial arrangement described in the prompt:
        - Keep all windows, doors, skylights in their exact positions
        - Keep all cabinets, counters, appliances in their exact positions
        - Keep the same room dimensions and proportions
        - Keep the same camera angle/perspective
        - ONLY change surface finishes: paint colors, cabinet colors, countertop materials, flooring, backsplash, hardware, and decorative elements
        - DO NOT move, add, or remove any structural elements or major fixtures
        
        **Use the SLC Formula for Photorealism:**
        
        1. **SUBJECT (S)** - Be highly specific about details and textures:
           - Describe exact materials with rich adjectives (e.g., "smooth matte white shaker-style cabinets", "honed Carrara marble countertops with subtle grey veining")
           - Include texture details (e.g., "brushed nickel hardware", "wide-plank oak flooring with natural grain")
           - Specify finishes precisely (e.g., "satin finish", "polished", "matte", "textured")
        
        2. **LIGHTING (L)** - Describe lighting conditions that create mood and realism:
           - Natural light sources (e.g., "soft morning sunlight streaming through windows", "golden hour warm glow")
           - Artificial lighting (e.g., "warm LED under-cabinet lighting", "pendant lights casting gentle shadows")
           - Light quality (e.g., "diffused natural light", "dramatic side lighting", "even ambient illumination")
           - Shadows and highlights (e.g., "subtle shadows adding depth", "highlights on polished surfaces")
        
        3. **CAMERA (C)** - Include professional photography specifications:
           - Camera type: "shot on professional DSLR" or "architectural photography camera"
           - Resolution: "8K resolution", "ultra high definition", "HDR"
           - Perspective: specific angle (e.g., "wide-angle lens from doorway", "eye-level perspective", "slightly elevated view")
           - Depth of field: "sharp focus throughout" or "shallow depth of field with background blur"
           - Quality keywords: "professional interior design photography", "magazine quality", "architectural digest style"
        
        **Additional Requirements:**
        - Maintain existing spatial layout and dimensions exactly as described
        - Include specific color names and codes when mentioned
        - Add atmospheric details (e.g., "clean, inviting atmosphere", "modern luxury feel")
        - Specify the aspect ratio: {inputs.aspect_ratio}
        
        **Output Format:** Create a single, flowing paragraph that reads like a professional photography brief. 
        Start with the camera/technical specs, then describe the subject with rich detail, then lighting conditions.
        Include keywords: "photorealistic", "8K", "HDR", "professional interior photography", "architectural photography".
        Emphasize that the layout must remain unchanged - only surface finishes are modified.
        """
        
        if reference_images:
            base_rewrite_prompt += "\n\n**Reference Image Layout:** The reference image shows the EXACT layout that must be preserved. Match the camera angle, room structure, window/door positions, and furniture/appliance placement EXACTLY. Only change the surface finishes and colors. Analyze the lighting in the reference image and replicate it."
        
        rewritten_prompt_response = client.models.generate_content(
            model="gemini-3-pro-preview", 
            contents=base_rewrite_prompt
        )
        rewritten_prompt = rewritten_prompt_response.text
        logger.info(f"Enhanced prompt: {rewritten_prompt}")

        content_parts = [types.Part.from_text(text=rewritten_prompt)]
        content_parts.extend(reference_images)
        contents = [
            types.Content(
                role="user",
                parts=content_parts,
            ),
        ]

        version = get_next_version_number(tool_context, inputs.asset_name)
        artifact_filename = create_versioned_filename(inputs.asset_name, version)
        logger.info(f"Generating rendering with artifact filename: {artifact_filename} (version {version})")

        return await _stream_and_save_rendering(
            client,
            contents,
            tool_context,
            inputs.asset_name,
            artifact_filename,
            version,
            success=(
                f"✅ Renovation rendering generated successfully!\n\n"
                f"The rendering has been saved and is available in the artifacts panel. "
                f"Artifact name: {inputs.asset_name} (version {version}).\n\n"
                f"Note: The image is stored as an artifact and can be accessed through the session artifacts, not as a direct image link."
            ),
            save_error="Error saving rendering as artifact",
            empty="No rendering was generated. Please try again with a more detailed prompt.",
        )
        
    except Exception as e:
        logger.error(f"Error in generate_renovation_rendering: {e}")
        return f"An error occurred while generating the rendering: {e}"


async def edit_renovation_rendering(tool_context: ToolContext, inputs: EditRenovationRenderingInput) -> str:
    """
    Edits an existing renovation rendering based on feedback or refinements.
    
    This tool allows iterative improvements to the rendered image, such as 
    changing colors, materials, lighting, or layout elements.
    """
    if "GEMINI_API_KEY" not in os.environ and "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.")

    logger.info("Starting renovation rendering edit")

    try:
        client = genai.Client()
        
        # ADK may pass a dict instead of the Pydantic model.
        if isinstance(inputs, dict):
            inputs = EditRenovationRenderingInput(**inputs)
        
        artifact_filename = inputs.artifact_filename
        if not artifact_filename:
            artifact_filename = tool_context.state.get("last_generated_rendering")
            if not artifact_filename:
                return "❌ No artifact_filename provided and no previous rendering found in session. Please generate a rendering first using generate_renovation_rendering."
            logger.info(f"Using last generated rendering from session: {artifact_filename}")
        
        # First version is always v1; models sometimes hallucinate _v0.
        if "_v0." in artifact_filename:
            logger.warning(f"Invalid version v0 detected in filename: {artifact_filename}")
            corrected_filename = artifact_filename.replace("_v0.", "_v1.")
            logger.info(f"Attempting corrected filename: {corrected_filename}")
            artifact_filename = corrected_filename
        
        logger.info(f"Loading artifact: {artifact_filename}")
        loaded_image_part = None
        try:
            loaded_image_part = await tool_context.load_artifact(artifact_filename)
        except Exception as e:
            logger.error(f"Error loading artifact: {e}")
        
        # Fallback: known filename for this asset, then last_generated_rendering.
        if not loaded_image_part:
            base_name = artifact_filename.split('_v')[0] if '_v' in artifact_filename else artifact_filename.replace('.png', '')
            asset_filenames = tool_context.state.get("asset_filenames", {})
            
            if base_name in asset_filenames:
                fallback_filename = asset_filenames[base_name]
                logger.info(f"Attempting fallback to known artifact: {fallback_filename}")
                try:
                    loaded_image_part = await tool_context.load_artifact(fallback_filename)
                    if loaded_image_part:
                        artifact_filename = fallback_filename
                        logger.info(f"Successfully loaded fallback artifact: {fallback_filename}")
                except Exception as e:
                    logger.error(f"Fallback load also failed: {e}")
            
            if not loaded_image_part:
                last_rendering = tool_context.state.get("last_generated_rendering")
                if last_rendering and last_rendering != artifact_filename:
                    logger.info(f"Attempting last resort fallback to: {last_rendering}")
                    try:
                        loaded_image_part = await tool_context.load_artifact(last_rendering)
                        if loaded_image_part:
                            artifact_filename = last_rendering
                            logger.info(f"Successfully loaded last resort artifact: {last_rendering}")
                    except Exception as e:
                        logger.error(f"Last resort load also failed: {e}")
        
        if not loaded_image_part:
            available_renderings = get_asset_versions_info(tool_context)
            return f"❌ Could not find rendering artifact: {inputs.artifact_filename}\n\n{available_renderings}\n\nPlease use one of the available artifact filenames, or generate a new rendering first."

        reference_image_part = None
        if inputs.reference_image_filename:
            reference_image_part = await load_reference_image(tool_context, inputs.reference_image_filename)
            if reference_image_part:
                logger.info(f"Using reference image for editing: {inputs.reference_image_filename}")

        content_parts = [loaded_image_part, types.Part.from_text(text=inputs.prompt)]
        if reference_image_part:
            content_parts.append(reference_image_part)

        contents = [
            types.Content(
                role="user",
                parts=content_parts,
            ),
        ]

        if inputs.asset_name:
            asset_name = inputs.asset_name
        else:
            current_asset_name = tool_context.state.get("current_asset_name")
            if current_asset_name:
                asset_name = current_asset_name
            else:
                base_name = artifact_filename.split('_v')[0] if '_v' in artifact_filename else "renovation_rendering"
                asset_name = base_name
        
        version = get_next_version_number(tool_context, asset_name)
        edited_artifact_filename = create_versioned_filename(asset_name, version)
        logger.info(f"Editing rendering with artifact filename: {edited_artifact_filename} (version {version})")

        return await _stream_and_save_rendering(
            client,
            contents,
            tool_context,
            asset_name,
            edited_artifact_filename,
            version,
            success=(
                f"✅ Rendering edited successfully!\n\n"
                f"The updated rendering has been saved and is available in the artifacts panel. "
                f"Artifact name: {asset_name} (version {version}).\n\n"
                f"Note: The image is stored as an artifact and can be accessed through the session artifacts, not as a direct image link."
            ),
            save_error="Error saving edited rendering as artifact",
            empty="No edited rendering was generated. Please try again.",
        )
        
    except Exception as e:
        logger.error(f"Error in edit_renovation_rendering: {e}")
        return f"An error occurred while editing the rendering: {e}"


async def list_renovation_renderings(tool_context: ToolContext) -> str:
    """Lists all renovation renderings created in this session."""
    return get_asset_versions_info(tool_context)
