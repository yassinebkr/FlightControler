import os
import requests
from bs4 import BeautifulSoup

# Base URL for the website
BASE_URL = 'https://docs.pyvista.org'

# Starting point - main examples page
START_URL = f'{BASE_URL}/examples/'

# Function to fetch the content of a page
def fetch_page(url):
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    else:
        print(f"Failed to retrieve {url}")
        return None

# Function to scrape the examples page and extract all relevant links
def scrape_examples_page(url):
    content = fetch_page(url)
    if content:
        soup = BeautifulSoup(content, 'html.parser')
        
        # Find all sections (links to subcategories)
        links = soup.find_all('a', class_='reference internal')
        examples = []

        for link in links:
            example_title = link.find('span', class_='std std-ref')
            if example_title:
                full_url = BASE_URL + link['href']
                examples.append((example_title.get_text(), full_url))
        
        return examples
    return []

# Function to scrape the content of an individual example page
def scrape_example_content(example_url):
    content = fetch_page(example_url)
    if content:
        soup = BeautifulSoup(content, 'html.parser')
        
        # Check if there's a specific div for the content (could vary depending on the page)
        content_div = soup.find('div', class_='wy-content')
        
        # If the content div exists, extract the text and code
        if content_div:
            text = content_div.get_text(separator='\n', strip=True)
            # You can also refine this to extract specific parts like code blocks if needed
            return text
        else:
            print(f"No content found on {example_url}")
    return None

# Save content to a single file
def save_to_file(examples):
    with open('pyvista_examples.txt', 'w', encoding='utf-8') as f:
        for title, url in examples:
            print(f"Scraping {title}...")
            f.write(f"Example: {title}\n")
            content = scrape_example_content(url)
            if content:
                f.write(content + "\n")
            else:
                f.write("No content found.\n")
            f.write("\n" + "="*80 + "\n")

# Main function
def main():
    print("Starting to scrape PyVista examples...\n")
    examples = scrape_examples_page(START_URL)
    print(f"Found {len(examples)} examples.\n")
    save_to_file(examples)
    print("Scraping complete. Saved to pyvista_examples.txt.")

# Run the script
if __name__ == "__main__":
    main()
