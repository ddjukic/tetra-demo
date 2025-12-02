import asyncio
import os
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

from clients.pubmed_client import PubMedClient

async def test_pubmed_and_stats():
    """Test PubMed client and collect token statistics."""
    client = PubMedClient()

    # Search for orexin papers after 2010
    query = "orexin AND 2010:2024[pdat]"
    print(f"Searching: {query}")

    pmids = await client.search(query, max_results=200)
    print(f"Found {len(pmids)} PMIDs")

    if not pmids:
        print("ERROR: No PMIDs found!")
        return

    # Fetch abstracts
    print(f"Fetching abstracts for {len(pmids)} papers...")
    articles = await client.fetch_abstracts(pmids)
    print(f"Fetched {len(articles)} articles with abstracts")

    # Calculate token statistics
    import tiktoken
    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except:
        # Fallback: estimate 4 chars per token
        enc = None

    abstract_lengths = []
    token_counts = []

    for article in articles:
        abstract = article.get('abstract', '')
        if abstract:
            abstract_lengths.append(len(abstract))
            if enc:
                tokens = len(enc.encode(abstract))
            else:
                tokens = len(abstract) // 4
            token_counts.append(tokens)

    if token_counts:
        import statistics
        print(f"\n=== Abstract Statistics ({len(token_counts)} abstracts) ===")
        print(f"Character lengths:")
        print(f"  Min: {min(abstract_lengths)}")
        print(f"  Max: {max(abstract_lengths)}")
        print(f"  Mean: {statistics.mean(abstract_lengths):.0f}")
        print(f"  Median: {statistics.median(abstract_lengths):.0f}")
        print(f"\nToken counts (cl100k_base or estimate):")
        print(f"  Min: {min(token_counts)}")
        print(f"  Max: {max(token_counts)}")
        print(f"  Mean: {statistics.mean(token_counts):.0f}")
        print(f"  Median: {statistics.median(token_counts):.0f}")
        print(f"  Total: {sum(token_counts)}")
        print(f"\nBatching estimates:")
        print(f"  All 200 abstracts: ~{sum(token_counts)} tokens")
        print(f"  At 10K tokens/batch: ~{sum(token_counts)//10000 + 1} batches needed")
        print(f"  At 32K tokens/batch: ~{sum(token_counts)//32000 + 1} batches needed")
    else:
        print("ERROR: No abstracts found in articles!")

asyncio.run(test_pubmed_and_stats())
