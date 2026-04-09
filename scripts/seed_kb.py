import sys
import os
import json
from tqdm import tqdm

# Add parent directory to path to allow importing from services
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.supabase_service import supabase_service
from services.embed_service import embed_service

# Sample PM Theory Data
SEED_DATA = [
    {
        "concept": "Risk Management",
        "content": "Risk management is the process of identifying, assessing, and controlling threats to an organization's capital and earnings. It involves risk identification, qualitative and quantitative analysis, risk response planning, and risk monitoring.",
        "metadata": {"source": "pmbok", "category": "theory"}
    },
    {
        "concept": "Scope Management",
        "content": "Project scope management includes the processes required to ensure that the project includes all the work required, and only the work required, to complete the project successfully. It involves defining and controlling what is and is not included.",
        "metadata": {"source": "pmbok", "category": "theory"}
    },
    {
        "concept": "Earned Value Management (EVM)",
        "content": "Earned Value Management is a project management technique for measuring project performance and progress. It combines measurements of project scope, schedule, and cost in a single integrated system. Key metrics include CPI and SPI.",
        "metadata": {"source": "pmbok", "category": "theory"}
    },
    {
        "concept": "Critical Path Method (CPM)",
        "content": "The Critical Path Method is a scheduling technique that identifies the longest sequence of tasks that must be finished on time for the entire project to finish on time. Delaying a task on the critical path delays the whole project.",
        "metadata": {"source": "pmbok", "category": "theory"}
    },
    {
        "concept": "Stakeholder Management",
        "content": "Stakeholder management involves identifying people, groups, or organizations that could impact or be impacted by a project, and developing strategies to effectively engage them throughout the project lifecycle.",
        "metadata": {"source": "pmbok", "category": "theory"}
    }
]

def seed_knowledge_base():
    """
    Embeds and uploads initial PM theory to Supabase.
    """
    if not supabase_service.client:
        print("[ERROR] Supabase client not initialized. Check your credentials.")
        return

    print(f"--- Seeding Knowledge Base with {len(SEED_DATA)} concepts ---")
    
    for item in tqdm(SEED_DATA):
        try:
            # 1. Generate Embedding
            text_to_embed = f"{item['concept']}: {item['content']}"
            embedding = embed_service.embed_text(text_to_embed)
            
            # 2. Insert into Supabase
            data = {
                "content": item['content'],
                "embedding": embedding,
                "metadata": {**item['metadata'], "concept": item['concept']}
            }
            
            supabase_service.client.table("knowledge_base").insert(data).execute()
        except Exception as e:
            print(f"[ERROR] Failed to seed concept '{item['concept']}': {e}")

if __name__ == "__main__":
    seed_knowledge_base()
    print("--- Seeding Complete ---")
