"""
Example: Using incremental save feature
This shows how to use the save_progress flag to monitor scraping in real-time
"""

from enhanced_web_scraper import EnhancedCompanyScraper
import os


def example_with_progress_saving():
    """
    Example: Scrape with incremental progress saving
    """
    print("="*80)
    print("Example: Incremental Progress Saving")
    print("="*80)
    print("\nThis example demonstrates:")
    print("  1. Data is saved after EACH brand is scraped")
    print("  2. You can monitor progress in real-time")
    print("  3. Data is safe even if scraping is interrupted")
    print("\n" + "="*80 + "\n")

    # Initialize scraper with save_progress=True
    company_name = "PepsiCo"
    scraper = EnhancedCompanyScraper(verbose=True, save_progress=True)

    print(f"Starting scrape for: {company_name}\n")
    print("💡 TIP: Open another terminal and run:")
    print(f"   watch -n 2 'ls -lh data/{company_name}*'")
    print("   to see files updating in real-time!\n")
    print("="*80 + "\n")

    # Perform scraping
    brands_df, esg_df = scraper.scrape_company_brands_and_esg(company_name)

    # Export final versions
    if not brands_df.empty:
        scraper.export_to_csv(brands_df, esg_df, company_name)
        scraper.create_combined_export(brands_df, esg_df, company_name)

        print("\n" + "="*80)
        print("✅ Complete!")
        print("="*80)
        print("\nFiles created:")

        # List all files
        data_dir = "data"
        company_clean = company_name.replace(' ', '_').replace('.', '')
        files = [f for f in os.listdir(data_dir) if f.startswith(company_clean)]

        for f in sorted(files):
            size = os.path.getsize(os.path.join(data_dir, f))
            print(f"  📄 {f} ({size:,} bytes)")

        progress_files = [f for f in files if 'progress' in f]
        final_files = [f for f in files if 'progress' not in f]

        print(f"\n📊 Summary:")
        print(f"  - Progress files (updated incrementally): {len(progress_files)}")
        print(f"  - Final files (complete data): {len(final_files)}")
        print(f"  - Total brands scraped: {len(brands_df)}")
        print("="*80)


def example_without_progress_saving():
    """
    Example: Standard scraping (no incremental saving)
    """
    print("="*80)
    print("Example: Standard Scraping (No Progress Saving)")
    print("="*80)
    print("\nThis example demonstrates standard mode:")
    print("  - Files are created only at the END")
    print("  - Faster (no file I/O during scraping)")
    print("  - Good for small brand lists")
    print("\n" + "="*80 + "\n")

    # Initialize scraper with save_progress=False (default)
    company_name = "PepsiCo"
    scraper = EnhancedCompanyScraper(verbose=True, save_progress=False)

    # Perform scraping
    brands_df, esg_df = scraper.scrape_company_brands_and_esg(company_name)

    # Export at the end
    if not brands_df.empty:
        scraper.export_to_csv(brands_df, esg_df, company_name)
        scraper.create_combined_export(brands_df, esg_df, company_name)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "standard":
        example_without_progress_saving()
    else:
        example_with_progress_saving()

        print("\n💡 To see standard mode without progress saving:")
        print("   python example_incremental_save.py standard")
