"""AI Home Renovation Planner - Coordinator/Dispatcher Pattern with Multimodal Vision

A routing agent analyzes requests and delegates to specialists:

- General questions → Info agent
- Renovation planning → Single planner (assess + design + render)
- Rendering edits → Rendering editor

Pattern Reference: https://google.github.io/adk-docs/agents/multi-agents/#coordinator-dispatcher-pattern
"""

from google.adk.agents import LlmAgent
from google.adk.tools import google_search
from google.adk.tools.agent_tool import AgentTool
from .tools import (
    generate_renovation_rendering,
    edit_renovation_rendering,
    list_renovation_renderings,
)


# ============================================================================
# Helper Tool Agent (wraps google_search)
# ============================================================================

# google_search is a pre-built tool function that allows the agent to perform Google searches
# Note: google_search can only be used by itself within an agent instance (single tool limitation)

search_agent = LlmAgent(
    name="SearchAgent",
    model="gemini-3-flash-preview",  # google_search requires Gemini 2.0+ models
    description="Searches for renovation costs, contractors, materials, and design trends",
    instruction="""You are a search specialist. When asked to find information about renovation costs, 
contractors, materials, or design trends, the search capability is automatically enabled. 
Simply respond with the information you find. Be concise and cite sources when available.""",
    tools=[google_search],
)


# ============================================================================
# Utility Tools
# ============================================================================

def estimate_renovation_cost(
    room_type: str,
    scope: str,
    square_footage: int,
) -> str:
    """Estimate renovation costs based on room type and scope.
    
    Args:
        room_type: Type of room (kitchen, bathroom, bedroom, living_room, etc.)
        scope: Renovation scope (cosmetic, moderate, full, luxury)
        square_footage: Room size in square feet
    
    Returns:
        Estimated cost range
    """
    # Cost per sq ft estimates (2024 ranges)
    rates = {
        "kitchen": {"cosmetic": (50, 100), "moderate": (150, 250), "full": (300, 500), "luxury": (600, 1200)},
        "bathroom": {"cosmetic": (75, 125), "moderate": (200, 350), "full": (400, 600), "luxury": (800, 1500)},
        "bedroom": {"cosmetic": (30, 60), "moderate": (75, 150), "full": (150, 300), "luxury": (400, 800)},
        "living_room": {"cosmetic": (40, 80), "moderate": (100, 200), "full": (200, 400), "luxury": (500, 1000)},
    }
    
    room = room_type.lower().replace(" ", "_")
    scope_level = scope.lower()
    
    if room not in rates:
        room = "living_room"
    if scope_level not in rates[room]:
        scope_level = "moderate"
    
    low, high = rates[room][scope_level]
    
    total_low = low * square_footage
    total_high = high * square_footage
    
    return f"💰 Estimated Cost: ${total_low:,} - ${total_high:,} ({scope_level} {room_type} renovation, ~{square_footage} sq ft)"


def calculate_timeline(
    scope: str,
) -> str:
    """Estimate renovation timeline based on scope.
    
    Args:
        scope: Renovation scope (cosmetic, moderate, full, luxury)
    
    Returns:
        Estimated timeline with phases
    """
    timelines = {
        "cosmetic": "1-2 weeks (quick refresh)",
        "moderate": "3-6 weeks (includes some structural work)",
        "full": "2-4 months (complete transformation)",
        "luxury": "4-6 months (custom work, high-end finishes)"
    }
    
    scope_level = scope.lower()
    timeline = timelines.get(scope_level, timelines["moderate"])
    
    return f"⏱️ Estimated Timeline: {timeline}"


# ============================================================================
# Specialist Agent 1: Info Agent (for general inquiries)
# ============================================================================

info_agent = LlmAgent(
    name="InfoAgent",
    model="gemini-3-flash-preview",
    description="Handles general renovation questions and provides system information",
    instruction="""
You are the Info Agent for the AI Home Renovation Planner.

WHEN TO USE: The coordinator routes general questions and casual greetings to you.

YOUR RESPONSE:
- Keep it brief and helpful (2-4 sentences)
- Explain the system helps with home renovations using visual AI
- Mention capabilities: photo analysis, design planning, budget estimation, timeline coordination
- Ask about their renovation project (which room, can they share photos?)

EXAMPLE:
"Hi! I'm your AI Home Renovation Planner. I can analyze photos of your current space and inspiration images to create a personalized renovation plan with design suggestions, budget estimates, and timelines. Which room are you thinking of renovating? Feel free to share photos if you have them!"

Be enthusiastic about home improvement and helpful!
""",
)


# ============================================================================
# Specialist Agent 2: Rendering Editor (for iterative refinements)
# ============================================================================

rendering_editor = LlmAgent(
    name="RenderingEditor",
    model="gemini-3-flash-preview",
    description="Edits existing renovation renderings based on user feedback",
    instruction="""
You refine existing renovation renderings.

**TASK**: User wants to modify an existing rendering (e.g., "make cabinets cream", "darker flooring").

**CRITICAL**: Find the most recent rendering filename from conversation history!
Look for: "Saved as artifact: [filename]" or "kitchen_modern_renovation_v1.png" type references.

Use **edit_renovation_rendering** tool:

Parameters:
1. artifact_filename: The exact filename of the most recent rendering
2. prompt: Very specific edit instruction (be detailed!)
3. asset_name: Base name without _vX (e.g., "kitchen_modern_renovation")

**Example:**
User: "Make the cabinets cream instead of white"
Last rendering: "kitchen_modern_renovation_v1.png"

Call: edit_renovation_rendering(
  artifact_filename="kitchen_modern_renovation_v1.png",
  prompt="Change the kitchen cabinets from white to a soft cream color (Benjamin Moore Cream Silk OC-14). Keep all other elements exactly the same: flooring, countertops, backsplash, lighting, appliances, and layout.",
  asset_name="kitchen_modern_renovation"
)

Be SPECIFIC in prompts - vague = poor results!

After editing, briefly confirm the change.

**IMPORTANT - DO NOT use markdown image syntax!**
- Do NOT output `![image](filename.png)` or similar markdown image links
- Simply confirm the edit was successful and mention the artifact is available in the artifacts panel
""",
    tools=[edit_renovation_rendering, list_renovation_renderings],
)


# ============================================================================
# Specialist: Renovation Planner (assess + design + render)
# ============================================================================
# Flattened from a 3-hop SequentialAgent. That pipeline never set output_key and
# never templated {key} state, so DesignPlanner/ProjectCoordinator were hoping
# transcript survived. One LlmAgent keeps image context and tools in a single turn.

renovation_planner = LlmAgent(
    name="RenovationPlanner",
    model="gemini-3-flash-preview",
    description="Analyzes room photos, produces a renovation plan (design, budget, timeline), and generates a photorealistic rendering",
    instruction="""
You are the renovation planner. Analyze the request (and any uploaded photos), produce a concrete plan, then generate a rendering.

## 1. Assess the space
Detect whether each image is a CURRENT ROOM or an INSPIRATION/STYLE reference.
- Current room: type, size if visible, condition, existing style, problems, opportunities.
- Document EXACT layout to preserve: windows, doors, cabinet configuration, appliance positions, sink, counters, special features, camera angle.
- Inspiration: style name, palette, materials, notable features.
- If both: compare and list surface-finish changes needed to match the inspiration.
- If only current room: suggest 2-3 style directions.
- If a budget is mentioned, call estimate_renovation_cost with room type, scope, and square footage (estimate from the photo or description).
- Use SearchAgent for current material/labor costs when needed.

## 2. Design (surface finishes ONLY)
KEEP THE EXACT LAYOUT. Do not move appliances, reconfigure cabinets, add/remove windows or doors, change footprint, or add/remove islands.
Specify only surface finishes on the existing layout:
- Cabinet color (paint name + code on existing cabinets)
- Wall color
- Countertops (material/color on existing counters)
- Flooring (same floor area)
- Backsplash (same wall areas)
- Hardware
- Lighting (replace in place; under-cabinet OK)
- Appliances: keep or replace like-for-like in the SAME locations

If a budget exists, separate must-haves vs nice-to-haves.
Call calculate_timeline with scope (cosmetic/moderate/full/luxury). Prefer cosmetic or moderate — no structural changes.

## 3. Deliver a scannable plan
**Budget Breakdown**: materials, labor, permits/fees, 10% contingency, total. If a budget was given, say whether it fits or suggest phasing.
**Timeline**: phases
**Contractors**: trades needed
**Design Summary**: tight bullets of the finish specs (product names and color codes)
**Action Checklist**: numbered next steps

## 4. Generate a rendering
Call generate_renovation_rendering with:
- prompt: a detailed description of the renovated room — preserved layout (windows, doors, cabinets, appliances, camera angle) plus the exact surface finishes (colors, materials, textures). Do not wrap it in camera/SLC photography jargon; the tool rewrites the prompt for photorealism.
- aspect_ratio: "16:9"
- asset_name: a slug like "kitchen_modern_farmhouse_renovation"

After generating, briefly (2-3 sentences) describe what the rendering shows.

Do NOT output markdown image syntax like `![image](filename.png)`. Mention that the artifact is available in the artifacts panel.
""",
    tools=[
        AgentTool(search_agent),
        estimate_renovation_cost,
        calculate_timeline,
        generate_renovation_rendering,
    ],
)


# ============================================================================
# Coordinator/Dispatcher (Root Agent)
# ============================================================================

root_agent = LlmAgent(
    name="HomeRenovationPlanner",
    model="gemini-3-flash-preview",
    description="Intelligent coordinator that routes renovation requests to the appropriate specialist. Supports image analysis!",
    instruction="""
You are the Coordinator for the AI Home Renovation Planner.

YOUR ROLE: Analyze the user's request and route it to the right specialist using transfer_to_agent.

ROUTING LOGIC:

1. **For general questions/greetings**:
   → transfer_to_agent to "InfoAgent"
   → Examples: "hi", "what do you do?", "how much do renovations cost?"

2. **For editing EXISTING renderings** (only if rendering was already generated):
   → transfer_to_agent to "RenderingEditor"
   → Examples: "make cabinets cream", "darker", "change color", "add lights"
   → User wants to MODIFY an existing rendering
   → Check: Was a rendering generated earlier?

3. **For NEW renovation planning**:
   → transfer_to_agent to "RenovationPlanner"
   → Examples: "Plan my kitchen", "Here's my space [photos]", "Help renovate"
   → First-time planning or new project
   → ALWAYS route here if images uploaded!

CRITICAL: You MUST use transfer_to_agent - don't answer directly!

Decision flow:
- Rendering exists + wants changes → RenderingEditor
- New project/images → RenovationPlanner
- Just chatting → InfoAgent

Be a smart router - match intent!
""",
    sub_agents=[
        info_agent,
        rendering_editor,
        renovation_planner,
    ],
)


__all__ = ["root_agent"]