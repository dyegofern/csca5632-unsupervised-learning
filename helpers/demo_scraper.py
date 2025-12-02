"""
Quick demo of the enhanced web scraper with DDGS library
This demonstrates brand discovery only (faster than full scraping)
"""

from enhanced_web_scraper import EnhancedCompanyScraper
import time


def demo_brand_discovery(company_name: str):
    """
    Demo: Just discover brands owned by a company (no ESG data collection)

    Args:
        company_name: Name of the company
    """
    print("=" * 80)
    print(f"🔍 Brand Discovery Demo for: {company_name}")
    print("=" * 80)
    print("\nThis demo shows the brand discovery functionality using DDGS library")
    print("(ESG data collection is skipped for speed)\n")

    # Initialize scraper
    scraper = EnhancedCompanyScraper(verbose=True)

    # Discover brands (this uses DDGS)
    start_time = time.time()
    brands = scraper.discover_company_brands(company_name)
    elapsed = time.time() - start_time

    # Display results
    print("\n" + "=" * 80)
    print("📊 RESULTS")
    print("=" * 80)
    print(f"\nCompany: {company_name}")
    print(f"Time taken: {elapsed:.2f} seconds")
    print(f"Brands discovered: {len(brands)}")

    if brands:
        print("\n🏷️  Brands found:")
        print("-" * 80)
        for i, brand in enumerate(brands, 1):
            print(f"  {i:2d}. {brand}")
        print("-" * 80)
    else:
        print("\n⚠️  No brands discovered")

    print("\n" + "=" * 80)
    print("✅ Demo complete!")
    print("\nTo run full scraping (with ESG data), use:")
    print(f"  python run_scraper.py '{company_name}'")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        company_name = " ".join(sys.argv[1:])
    else:
        # Default demo
        company_name = "PepsiCo"
        print("\nNo company name provided. Using default: PepsiCo")
        print("Usage: python demo_scraper.py 'Company Name'\n")

    demo_brand_discovery(company_name)
