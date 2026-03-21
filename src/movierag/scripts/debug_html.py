from bs4 import BeautifulSoup
import json

file_path = "d:\\Study\\School\\project_ky4\\src\\imdb_dump.html"
output_path = "d:\\Study\\School\\project_ky4\\src\\debug_props.json"

with open(file_path, "r", encoding="utf-8") as f:
    html_content = f.read()

soup = BeautifulSoup(html_content, "html.parser")
script = soup.find("script", id="__NEXT_DATA__")

if script:
    try:
        data = json.loads(script.string)
        page_props = data.get("props", {}).get("pageProps", {})

        with open(output_path, "w", encoding="utf-8") as f_out:
            json.dump(page_props, f_out, indent=2)

        print(f"Successfully saved pageProps to {output_path}")

    except json.JSONDecodeError as e:
        print(f"JSON Error: {e}")
else:
    print("Script tag not found")
