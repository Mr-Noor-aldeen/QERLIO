import os
import sys
from playwright.sync_api import sync_playwright

def capture_dashboard_screenshot():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    html_file = os.path.join(base_dir, "server", "local_api", "dashboard.html")
    
    # Target folder specified by the user: "صورة hud"
    output_dir = os.path.join(base_dir, "صورة hud")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "qerlio_hud_dashboard.png")

    file_url = f"file:///{html_file.replace(os.sep, '/')}"
    print(f"Loading: {file_url}")

    with sync_playwright() as p:
        browser = None
        # Try default chromium or system installed chrome/msedge
        for channel in [None, "msedge", "chrome"]:
            try:
                kwargs = {"channel": channel} if channel else {}
                browser = p.chromium.launch(**kwargs)
                print(f"Launched browser with channel: {channel or 'default'}")
                break
            except Exception as e:
                print(f"Could not launch with channel {channel}: {e}")

        if not browser:
            raise RuntimeError("No suitable browser could be launched by Playwright.")

        page = browser.new_page(viewport={"width": 1440, "height": 920}, device_scale_factor=2)
        page.goto(file_url)
        page.wait_for_timeout(1000)  # Wait for JS to populate tables

        # Also let's populate some realistic stream in the chat output box for the screenshot
        page.evaluate("""
            document.getElementById('chat-output').textContent = 
                'User: Explain QERLIO adaptive on-device execution.\\n\\n' +
                'Assistant: QERLIO operates locally with zero cloud dependencies. ' +
                'It deploys an adaptive thermal and memory scheduler that monitors ' +
                'hardware temperature in real-time and dynamically scales KV-cache ' +
                'allocation to guarantee zero out-of-memory terminations.';
        """)

        page.screenshot(path=output_path, full_page=True)
        browser.close()

    print(f"Screenshot successfully saved to: {output_path}")

    # Also copy to Downloads folder for immediate user convenience
    downloads_hud_dir = os.path.join(os.path.expanduser("~"), "Downloads", "صورة hud")
    os.makedirs(downloads_hud_dir, exist_ok=True)
    downloads_hud_path = os.path.join(downloads_hud_dir, "qerlio_hud_dashboard.png")
    
    with open(output_path, "rb") as src, open(downloads_hud_path, "wb") as dst:
        dst.write(src.read())
    print(f"Also copied to user Downloads: {downloads_hud_path}")

if __name__ == "__main__":
    capture_dashboard_screenshot()
