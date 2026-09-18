# 🏚️ 🍌 AI Home Renovation Planner Agent 

### 🎓 FREE Step-by-Step Tutorial 
**👉 [Click here to follow our complete step-by-step tutorial](https://www.theunwindai.com/p/build-an-ai-home-renovation-planner-agent-using-nano-banana) and learn how to build this from scratch with detailed code walkthroughs, explanations, and best practices.**

A multi-agent system built with Google ADK that analyzes photos of your space, creates personalized renovation plans, and generates photorealistic renderings using Gemini 3 Flash and Gemini 3 Pro's multimodal capabilities.

## Features

- **🔍 Smart Image Analysis**: Upload room photos and inspiration images - agent automatically detects and analyzes them
- **🎨 Photorealistic Rendering**: Generates professional-quality images of your renovated space using Gemini 3 Pro
- **💰 Budget-Aware Planning**: Tailors recommendations to your budget constraints
- **📊 Complete Roadmap**: Provides timeline, budget breakdown, contractor list, and action checklist
- **🤖 Multi-Agent Orchestration**: Coordinator/Dispatcher routing to an info agent, a renovation planner, and a rendering editor
- **✏️ Iterative Refinement**: Edit generated renderings based on feedback

## How It Works

The system uses a **Coordinator/Dispatcher pattern**:

1. **Renovation Planner** 📸
   - Analyzes uploaded room photos and inspiration images
   - Specifies surface-finish-only design (layout preserved)
   - Estimates budget/timeline and generates a photorealistic rendering

2. **Rendering Editor** ✏️
   - Refines an existing rendering from feedback

3. **Info Agent** 💬
   - Handles greetings and general questions

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/mdyerapis-coder/ai-home-renovation-agent.git ai_home_renovation_agent
   cd ai_home_renovation_agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   ```bash
   export GOOGLE_API_KEY="your_gemini_api_key"
   ```
   Or create a `.env` file:
   ```
   GOOGLE_API_KEY=your_gemini_api_key
   ```

4. **Launch ADK Web** (from this repo root)
   ```bash
   adk web
   ```

5. **Open** http://localhost:8000 and select `ai_home_renovation_agent`

## Usage Examples

### Scenario 1: Current Room + Budget
```
[Upload photo of your kitchen]
"What can I improve here with a $5k budget?"
```
→ Agent analyzes your space, suggests budget-friendly improvements, generates rendering

### Scenario 2: Room + Inspiration
```
[Upload photo 1: your kitchen]
[Upload photo 2: Pinterest inspiration]
"Transform my kitchen to look like this. What's the cost?"
```
→ Agent extracts style from inspiration, applies to your room, provides budget + rendering

### Scenario 3: Text Only
```
"Renovate my 10x12 kitchen with oak cabinets and laminate counters. 
Want modern farmhouse style with white shaker cabinets. Budget: $30k"
```
→ Agent creates design plan and generates rendering from description

### Scenario 4: Iterative Refinement
```
[After initial rendering]
"Make the cabinets cream instead of white"
"Add pendant lights over the island"
"Change flooring to lighter oak"
```
→ Agent refines the rendering with your feedback

## Sample Prompts
- "I want to renovate my small galley kitchen. It's 8x12 feet, has oak cabinets from the 90s. I love modern farmhouse style. Budget: $25k"
- "My master bathroom is tiny (5x8) with a cramped tub. I want a spa-like retreat with walk-in shower. Budget: $15k"
- "Transform my boring bedroom into a cozy retreat. Thinking accent wall, new flooring. Budget: $12k"

## Tools & Capabilities

**RenovationPlanner** (assess + design + render)
- **SearchAgent**: Google search for costs, materials, and trends
- **estimate_renovation_cost**: Cost by room type and scope
- **calculate_timeline**: Duration by scope
- **generate_renovation_rendering**: Photorealistic rendering

**RenderingEditor**
- **edit_renovation_rendering**: Refine a rendering from feedback
- **list_renovation_renderings**: List versioned renderings in the session

## Multi-Agent Pattern

Demonstrates **Coordinator/Dispatcher**:

```
Coordinator (Root Agent)
    ├── Info Agent (quick Q&A)
    ├── Renovation Planner (assess + design + render)
    └── Rendering Editor (iterative image edits)
```

Planning is one LlmAgent (assess + design + render), not a SequentialAgent. SLC photorealism rewrite lives only in `generate_renovation_rendering`.

