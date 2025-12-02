"""
Simple script to run the enhanced web scraper for any company
"""

import sys
from enhanced_web_scraper import EnhancedCompanyScraper


def scrape_company(company_name: str, save_progress: bool = False):
    """
    Scrape brands and ESG data for a company

    Args:
        company_name: Name of the company (e.g., "PepsiCo", "Coca-Cola")
        save_progress: Save data incrementally after each brand (default: False)
    """
    print(f"\n{'='*80}")
    print(f"Enhanced Web Scraper - {company_name}")
    print(f"{'='*80}\n")

    # Initialize scraper
    scraper = EnhancedCompanyScraper(verbose=True, save_progress=save_progress)

    # Scrape data
    brands_df, esg_df = scraper.scrape_company_brands_and_esg(company_name)

    # Export if data was found
    if not brands_df.empty:
        # Export separate files
        brands_file, esg_file = scraper.export_to_csv(brands_df, esg_df, company_name)

        # Export combined file for database import
        combined_file = scraper.create_combined_export(brands_df, esg_df, company_name)

        print("\n" + "="*80)
        print("📊 SUMMARY")
        print("="*80)
        print(f"\nCompany: {company_name}")
        print(f"Brands discovered: {len(brands_df)}")
        print(f"Brands with data: {brands_df['scrape_success'].sum()}")
        print(f"Brands with ESG data: {esg_df['esg_data_found'].sum()}")

        print(f"\n📈 ESG METRICS:")
        print(f"  ├─ CO2/Emissions mentions: {esg_df['co2_emissions_mentioned'].sum()}/{len(esg_df)}")
        print(f"  ├─ Pollution mentions: {esg_df['pollution_mentioned'].sum()}/{len(esg_df)}")
        print(f"  ├─ Carbon neutral claims: {esg_df['carbon_neutral_claim'].sum()}/{len(esg_df)}")
        print(f"  └─ Sustainability reports: {esg_df['sustainability_report_found'].sum()}/{len(esg_df)}")

        print(f"\n📁 FILES CREATED:")
        print(f"  ├─ Brands data: {brands_file}")
        print(f"  ├─ ESG data: {esg_file}")
        print(f"  └─ Combined (database-ready): {combined_file}")

        print("\n" + "="*80)
        print("✅ Scraping completed successfully!")
        print("="*80 + "\n")

        # Show sample of brands found
        if len(brands_df) > 0:
            print("\n🏷️  SAMPLE BRANDS FOUND:")
            print("-" * 80)
            sample_size = min(10, len(brands_df))
            for idx, row in brands_df.head(sample_size).iterrows():
                success = "✓" if row['scrape_success'] else "✗"
                print(f"  {success} {row['brand_name']}")
                if row['category']:
                    print(f"    Category: {row['category']}")
            if len(brands_df) > sample_size:
                print(f"  ... and {len(brands_df) - sample_size} more brands")
            print("-" * 80)

    else:
        print("\n" + "="*80)
        print("⚠️  No data collected")
        print("="*80)
        print("\nPossible reasons:")
        print("  - Company name not recognized")
        print("  - Network connectivity issues")
        print("  - Rate limiting by search engines")
        print("\nSuggestions:")
        print("  - Verify the company name spelling")
        print("  - Try again after a few minutes")
        print("  - Check your internet connection")
        print("="*80 + "\n")


if __name__ == "__main__":
    # Check for --save-progress flag
    save_progress = '--save-progress' in sys.argv or '-p' in sys.argv

    # Remove flag from arguments
    args = [arg for arg in sys.argv[1:] if arg not in ['--save-progress', '-p']]

    # Check if company name provided via command line
    if len(args) > 0:
        company_name = " ".join(args)
        scrape_company(company_name, save_progress=save_progress)
    else:
        # Interactive mode
        print("="*80)
        print("Enhanced Company Brand & ESG Data Scraper")
        print("="*80)
        print("\nThis tool will:")
        print("  1. Discover all brands owned by a company")
        print("  2. Collect ESG/CO2/Pollution data for each brand")
        print("  3. Export data to CSV files for database import")
        print("\n" + "="*80)

        # Get company name from user
        company_name = input("\nEnter company name (e.g., PepsiCo, Coca-Cola): ").strip()

        # Ask about incremental saving
        save_choice = input("\nSave progress after each brand? (y/n) [default: n]: ").strip().lower()
        save_progress = save_choice in ['y', 'yes']

        if company_name:
            scrape_company(company_name, save_progress=save_progress)
        else:
            print("\n⚠️  No company name provided. Exiting.")
            print("\nUsage:")
            print("  Interactive: python run_scraper.py")
            print("  Command line: python run_scraper.py 'Company Name'")
            print("  With progress saving: python run_scraper.py 'Company Name' --save-progress")
            print("\nExamples:")
            print("  python run_scraper.py PepsiCo")
            print("  python run_scraper.py 'Coca-Cola' --save-progress")
            print("  python run_scraper.py Unilever -p")
