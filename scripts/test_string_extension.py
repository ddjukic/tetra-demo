#!/usr/bin/env python3
"""
Test STRING API extension capabilities in isolation.

This script tests the STRING client's network extension features:
1. Basic get_network() without extension (baseline)
2. get_network() with add_nodes parameter to expand the network
3. get_interaction_partners() to discover all partners
4. Compare statistics: with vs without extension

Run with: uv run python scripts/test_string_extension.py
"""

import asyncio
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from clients.string_client import StringClient


def extract_proteins_from_interactions(interactions: list[dict[str, Any]]) -> set[str]:
    """Extract unique protein names from interaction list."""
    proteins = set()
    for interaction in interactions:
        proteins.add(interaction.get("preferredName_A", ""))
        proteins.add(interaction.get("preferredName_B", ""))
    proteins.discard("")
    return proteins


def print_interaction_summary(
    interactions: list[dict[str, Any]],
    title: str
) -> None:
    """Print a summary of interactions."""
    proteins = extract_proteins_from_interactions(interactions)

    print(f"\n{title}")
    print("-" * 50)
    print(f"  Interactions: {len(interactions)}")
    print(f"  Unique proteins: {len(proteins)}")

    if proteins:
        print(f"  Proteins: {', '.join(sorted(proteins)[:15])}")
        if len(proteins) > 15:
            print(f"            ...and {len(proteins) - 15} more")

    # Show top interactions by score
    if interactions:
        sorted_interactions = sorted(
            interactions,
            key=lambda x: float(x.get("score", 0)),
            reverse=True
        )
        print(f"\n  Top 5 interactions by score:")
        for i, interaction in enumerate(sorted_interactions[:5], 1):
            score = float(interaction.get("score", 0))
            name_a = interaction.get("preferredName_A", "?")
            name_b = interaction.get("preferredName_B", "?")
            print(f"    {i}. {name_a} <-> {name_b} (score: {score:.3f})")


async def test_baseline_network(
    client: StringClient,
    seed_proteins: list[str],
    min_score: int = 700,
) -> list[dict[str, Any]]:
    """Test basic network without extension (baseline)."""
    print("\n" + "=" * 60)
    print("TEST 1: Baseline Network (no extension)")
    print("=" * 60)
    print(f"Seed proteins: {seed_proteins}")
    print(f"Min score: {min_score}")

    interactions = await client.get_network(
        proteins=seed_proteins,
        min_score=min_score,
        network_type="physical",
        add_nodes=0,  # No extension
    )

    print_interaction_summary(interactions, "Baseline Results")
    return interactions


async def test_extended_network(
    client: StringClient,
    seed_proteins: list[str],
    add_nodes: int = 10,
    min_score: int = 700,
) -> list[dict[str, Any]]:
    """Test network with add_nodes extension."""
    print("\n" + "=" * 60)
    print(f"TEST 2: Extended Network (add_nodes={add_nodes})")
    print("=" * 60)
    print(f"Seed proteins: {seed_proteins}")
    print(f"Min score: {min_score}")
    print(f"Additional nodes requested: {add_nodes}")

    interactions = await client.get_network(
        proteins=seed_proteins,
        min_score=min_score,
        network_type="physical",
        add_nodes=add_nodes,
    )

    print_interaction_summary(interactions, f"Extended Results (add_nodes={add_nodes})")
    return interactions


async def test_interaction_partners(
    client: StringClient,
    seed_proteins: list[str],
    limit: int = 20,
    min_score: int = 700,
) -> list[dict[str, Any]]:
    """Test get_interaction_partners for discovering all partners."""
    print("\n" + "=" * 60)
    print(f"TEST 3: Interaction Partners (limit={limit})")
    print("=" * 60)
    print(f"Seed proteins: {seed_proteins}")
    print(f"Min score: {min_score}")
    print(f"Partners per protein: {limit}")

    partners = await client.get_interaction_partners(
        proteins=seed_proteins,
        limit=limit,
        min_score=min_score,
    )

    print_interaction_summary(partners, f"Partners Results (limit={limit})")

    # Show which proteins are partners (not seeds)
    partner_proteins = extract_proteins_from_interactions(partners)
    seed_set = set(p.upper() for p in seed_proteins)
    new_partners = {p for p in partner_proteins if p.upper() not in seed_set}

    print(f"\n  New partners discovered: {len(new_partners)}")
    if new_partners:
        print(f"  Partners: {', '.join(sorted(new_partners)[:15])}")
        if len(new_partners) > 15:
            print(f"            ...and {len(new_partners) - 15} more")

    return partners


async def test_combined_strategy(
    client: StringClient,
    seed_proteins: list[str],
    add_nodes: int = 10,
    partner_limit: int = 10,
    min_score: int = 700,
) -> dict[str, Any]:
    """Test combined strategy: get partners first, then expand network."""
    print("\n" + "=" * 60)
    print("TEST 4: Combined Strategy")
    print("=" * 60)
    print(f"Seed proteins: {seed_proteins}")
    print(f"Strategy: Get top partners, then use add_nodes for connections")

    # Step 1: Get interaction partners
    print(f"\n  Step 1: Getting top {partner_limit} partners per seed...")
    partners = await client.get_interaction_partners(
        proteins=seed_proteins,
        limit=partner_limit,
        min_score=min_score,
    )

    # Extract partner proteins
    partner_proteins = extract_proteins_from_interactions(partners)
    seed_set = set(p.upper() for p in seed_proteins)
    new_partners = {p for p in partner_proteins if p.upper() not in seed_set}

    print(f"  Discovered {len(new_partners)} new partner proteins")

    # Step 2: Get expanded network with add_nodes
    print(f"\n  Step 2: Getting network with add_nodes={add_nodes}...")
    extended_interactions = await client.get_network(
        proteins=seed_proteins,
        min_score=min_score,
        network_type="physical",
        add_nodes=add_nodes,
    )

    extended_proteins = extract_proteins_from_interactions(extended_interactions)

    # Combine all unique proteins
    all_proteins = partner_proteins | extended_proteins | set(seed_proteins)

    print(f"\n  Combined Results:")
    print(f"    - From partners: {len(partner_proteins)} proteins")
    print(f"    - From extended network: {len(extended_proteins)} proteins")
    print(f"    - Total unique proteins: {len(all_proteins)}")
    print(f"    - Partner interactions: {len(partners)}")
    print(f"    - Extended network interactions: {len(extended_interactions)}")

    # List all proteins by type
    print(f"\n  Protein breakdown:")
    print(f"    - Seeds: {sorted(seed_proteins)}")
    added_proteins = sorted(all_proteins - set(seed_proteins))
    print(f"    - Added ({len(added_proteins)}): {', '.join(added_proteins[:20])}")
    if len(added_proteins) > 20:
        print(f"                  ...and {len(added_proteins) - 20} more")

    return {
        "seed_proteins": seed_proteins,
        "all_proteins": sorted(all_proteins),
        "partner_interactions": partners,
        "extended_interactions": extended_interactions,
        "total_proteins": len(all_proteins),
        "total_partner_interactions": len(partners),
        "total_extended_interactions": len(extended_interactions),
    }


async def compare_results(
    baseline: list[dict[str, Any]],
    extended: list[dict[str, Any]],
    partners: list[dict[str, Any]],
) -> None:
    """Compare and summarize all results."""
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)

    baseline_proteins = extract_proteins_from_interactions(baseline)
    extended_proteins = extract_proteins_from_interactions(extended)
    partner_proteins = extract_proteins_from_interactions(partners)

    print(f"\nMetric               | Baseline | Extended | Partners")
    print("-" * 60)
    print(f"Interactions         |    {len(baseline):>5} |    {len(extended):>5} |    {len(partners):>5}")
    print(f"Unique Proteins      |    {len(baseline_proteins):>5} |    {len(extended_proteins):>5} |    {len(partner_proteins):>5}")

    # Calculate expansion ratios
    if len(baseline) > 0:
        ext_ratio = len(extended) / len(baseline)
        part_ratio = len(partners) / len(baseline)
        print(f"\nExpansion Ratios (vs baseline):")
        print(f"  - Extended network: {ext_ratio:.1f}x interactions")
        print(f"  - Partners: {part_ratio:.1f}x interactions")

    # Show new proteins discovered
    new_from_ext = extended_proteins - baseline_proteins
    new_from_partners = partner_proteins - baseline_proteins

    print(f"\nNew proteins discovered:")
    print(f"  - From extension: {len(new_from_ext)} proteins")
    if new_from_ext:
        print(f"    {', '.join(sorted(new_from_ext)[:10])}")
    print(f"  - From partners: {len(new_from_partners)} proteins")
    if new_from_partners:
        print(f"    {', '.join(sorted(new_from_partners)[:10])}")


async def main() -> None:
    """Run all STRING extension tests."""
    print("=" * 60)
    print("STRING NETWORK EXTENSION TEST")
    print("=" * 60)
    print("\nTesting STRING API extension capabilities with orexin proteins")

    # Orexin pathway seed proteins
    seed_proteins = ["HCRTR1", "HCRTR2", "HCRT"]
    min_score = 700  # High confidence

    async with StringClient() as client:
        try:
            # Test 1: Baseline (no extension)
            baseline = await test_baseline_network(
                client, seed_proteins, min_score
            )

            # Test 2: Extended network with add_nodes
            extended = await test_extended_network(
                client, seed_proteins, add_nodes=10, min_score=min_score
            )

            # Test 3: Interaction partners
            partners = await test_interaction_partners(
                client, seed_proteins, limit=20, min_score=min_score
            )

            # Test 4: Combined strategy
            combined = await test_combined_strategy(
                client, seed_proteins,
                add_nodes=10,
                partner_limit=10,
                min_score=min_score
            )

            # Compare all results
            await compare_results(baseline, extended, partners)

            print("\n" + "=" * 60)
            print("ALL STRING EXTENSION TESTS COMPLETED SUCCESSFULLY")
            print("=" * 60)

            # Final recommendation
            print("\nRECOMMENDATION:")
            print("-" * 50)
            baseline_count = len(baseline)
            extended_count = len(extended)
            partners_count = len(partners)

            if extended_count > baseline_count:
                print(f"  Using add_nodes=10 expands from {baseline_count} to {extended_count} interactions")
                print(f"  ({(extended_count/baseline_count - 1)*100:.0f}% increase)")
            else:
                print("  Note: add_nodes may not be adding new proteins for this query")

            if partners_count > baseline_count:
                print(f"\n  get_interaction_partners() finds {partners_count} interactions")
                print(f"  This is best for discovering all related proteins")

        except Exception as e:
            print(f"\nTest failed with error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
