#!/usr/bin/env python
"""
Test script for INDRA/Gilda entity grounding integration.

This script:
1. Loads a knowledge graph from a GraphML file
2. Grounds all protein/gene entities using Gilda API
3. Deduplicates entities by HGNC ID
4. Reports before/after statistics and merged entities

Usage:
    uv run python scripts/test_gilda_grounding.py [path_to_graphml]

If no path is provided, uses the default orexin graph.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from clients.gilda_client import GildaClient
from models.knowledge_graph import KnowledgeGraph


async def run_grounding_test(graph_path: str) -> None:
    """Run the grounding and deduplication test.

    Args:
        graph_path: Path to the GraphML file to process.
    """
    print("=" * 70)
    print("INDRA/Gilda Entity Grounding Test")
    print("=" * 70)
    print()

    # Load graph
    print(f"Loading graph from: {graph_path}")
    kg = KnowledgeGraph.load(graph_path)

    # Get initial statistics
    initial_summary = kg.to_summary()
    initial_nodes = initial_summary["node_count"]
    initial_edges = initial_summary["edge_count"]

    print(f"\n--- INITIAL GRAPH STATISTICS ---")
    print(f"Total nodes: {initial_nodes}")
    print(f"Total edges: {initial_edges}")
    print(f"Entity types: {initial_summary['entity_types']}")

    # Get groundable entities count
    groundable_entities = kg.get_groundable_entities()
    print(f"\nGroundable entities (protein/gene): {len(groundable_entities)}")

    # List some example entities before grounding
    print(f"\nSample entities before grounding:")
    for entity_id in list(groundable_entities)[:10]:
        entity_data = kg.entities[entity_id]
        print(f"  - {entity_id}: {entity_data.get('name', 'N/A')} ({entity_data.get('type', 'N/A')})")

    # Ground entities
    print(f"\n--- GROUNDING ENTITIES ---")
    print("Calling INDRA/Gilda API...")

    async with GildaClient() as gilda:
        grounding_stats = await kg.ground_entities(gilda)

    print(f"\nGrounding Results:")
    print(f"  Total processed: {grounding_stats['total_entities']}")
    print(f"  Successfully grounded: {grounding_stats['grounded']}")
    print(f"  Failed to ground: {grounding_stats['ungrounded']}")
    print(f"  HGNC grounded: {grounding_stats['hgnc_grounded']}")

    # Show grounding details
    grounding_map = grounding_stats["grounding_map"]
    grounded_entities = [(k, v) for k, v in grounding_map.items() if v is not None]
    ungrounded_entities = [k for k, v in grounding_map.items() if v is None]

    print(f"\n--- GROUNDED ENTITIES (showing first 20) ---")
    for entity_id, grounding in list(grounded_entities)[:20]:
        entry_name = grounding["entry_name"]
        full_id = grounding["full_id"]
        score = grounding["score"]
        print(f"  {entity_id} -> {full_id} ({entry_name}) [score: {score:.3f}]")

    if len(grounded_entities) > 20:
        print(f"  ... and {len(grounded_entities) - 20} more")

    print(f"\n--- UNGROUNDED ENTITIES ---")
    for entity_id in ungrounded_entities[:15]:
        print(f"  - {entity_id}")
    if len(ungrounded_entities) > 15:
        print(f"  ... and {len(ungrounded_entities) - 15} more")

    # Find potential duplicates (same HGNC ID)
    hgnc_groups: dict[str, list[str]] = {}
    for entity_id, data in kg.entities.items():
        hgnc_id = data.get("hgnc_id")
        if hgnc_id:
            if hgnc_id not in hgnc_groups:
                hgnc_groups[hgnc_id] = []
            hgnc_groups[hgnc_id].append(entity_id)

    duplicate_groups = {k: v for k, v in hgnc_groups.items() if len(v) > 1}

    print(f"\n--- DETECTED SYNONYM GROUPS (before deduplication) ---")
    print(f"Total HGNC groups: {len(hgnc_groups)}")
    print(f"Groups with duplicates: {len(duplicate_groups)}")

    if duplicate_groups:
        print(f"\nSynonym groups to merge:")
        for hgnc_id, entities in list(duplicate_groups.items())[:15]:
            print(f"  {hgnc_id}: {entities}")
        if len(duplicate_groups) > 15:
            print(f"  ... and {len(duplicate_groups) - 15} more")
    else:
        print("  No duplicate entities detected (all entity names are unique)")

    # Deduplicate
    print(f"\n--- DEDUPLICATING BY HGNC ID ---")
    dedup_stats = kg.deduplicate_by_hgnc()

    print(f"\nDeduplication Results:")
    print(f"  Original entity count: {dedup_stats['original_count']}")
    print(f"  Final entity count: {dedup_stats['final_count']}")
    print(f"  Entities removed: {dedup_stats['original_count'] - dedup_stats['final_count']}")
    print(f"  Synonym groups merged: {len(dedup_stats['merged_groups'])}")
    print(f"  Edges remapped: {dedup_stats['edges_remapped']}")

    if dedup_stats["merged_groups"]:
        print(f"\n--- MERGED GROUPS (detail) ---")
        for group in dedup_stats["merged_groups"][:10]:
            hgnc_id = group["hgnc_id"]
            canonical = group["canonical"]
            merged = group["merged"]
            print(f"  {hgnc_id}:")
            print(f"    Canonical: {canonical}")
            print(f"    Merged aliases: {merged}")
        if len(dedup_stats["merged_groups"]) > 10:
            print(f"  ... and {len(dedup_stats['merged_groups']) - 10} more groups")

    # Final statistics
    final_summary = kg.to_summary()
    grounding_summary = kg.get_grounding_summary()

    print(f"\n--- FINAL GRAPH STATISTICS ---")
    print(f"Total nodes: {final_summary['node_count']} (was {initial_nodes})")
    print(f"Total edges: {final_summary['edge_count']} (was {initial_edges})")
    print(f"Entity types: {final_summary['entity_types']}")

    print(f"\n--- GROUNDING SUMMARY ---")
    print(f"Total groundable: {grounding_summary['total_groundable']}")
    print(f"Grounded: {grounding_summary['grounded']}")
    print(f"HGNC grounded: {grounding_summary['hgnc_grounded']}")
    print(f"By database: {grounding_summary['by_database']}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    reduction = initial_nodes - final_summary["node_count"]
    reduction_pct = (reduction / initial_nodes * 100) if initial_nodes > 0 else 0
    print(f"Entity reduction: {reduction} nodes ({reduction_pct:.1f}%)")
    print(f"Synonym groups merged: {len(dedup_stats['merged_groups'])}")
    print(f"Grounding success rate: {grounding_stats['grounded']}/{grounding_stats['total_entities']} "
          f"({100 * grounding_stats['grounded'] / grounding_stats['total_entities']:.1f}%)"
          if grounding_stats['total_entities'] > 0 else "N/A")
    print()


def main() -> None:
    """Main entry point."""
    # Default graph path
    default_path = str(project_root / "graphs" / "orexin_singaling_20251205_004334.graphml")

    # Use provided path or default
    if len(sys.argv) > 1:
        graph_path = sys.argv[1]
    else:
        graph_path = default_path

    # Check if file exists
    if not Path(graph_path).exists():
        print(f"Error: Graph file not found: {graph_path}")
        print(f"\nAvailable graphs:")
        graphs_dir = project_root / "graphs"
        if graphs_dir.exists():
            for f in sorted(graphs_dir.glob("*.graphml")):
                print(f"  {f}")
        sys.exit(1)

    # Run test
    asyncio.run(run_grounding_test(graph_path))


if __name__ == "__main__":
    main()
