# Kenya 24-Hour Forecast Explorer

An interactive, browser-based explorer for viewing rainfall and temperature forecast values across Kenya's counties and wards.

## Features

- Interactive SVG map of Kenya with county boundaries.
- County and ward selection, including selecting a county directly on the map.
- Forecast parameter selection:
  - 24-hour cumulative rainfall
  - Maximum temperature
  - Minimum temperature
- Seven-day navigation with play/pause and animation-speed controls.
- Map zoom controls and a colour legend.
- Self-contained HTML page: the map, interface, JavaScript, and displayed data are embedded in `index.html`.

## Run locally

No package manager or build step is required.

1. Download or clone this repository.
2. Open `index.html` in a modern web browser.

Alternatively, from the project directory, run a simple local web server:

```bash
python3 -m http.server 8000
```

Then visit <http://localhost:8000>.

## Publish with GitHub Pages

1. Push this repository to GitHub.
2. Open the repository's **Settings → Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select the `main` branch and the `/ (root)` folder, then save.
5. Wait for GitHub Pages to publish the site. The public URL will usually look like:

   `https://YOUR-USERNAME.github.io/rainfall-explorer/`

Replace `YOUR-USERNAME` with your GitHub username.

## Important data note

The forecast values and map imagery/data are embedded in the HTML file. The page does **not** automatically fetch or refresh live forecasts. Its displayed dates and values may become outdated, so do not rely on it for operational weather, emergency, or safety decisions. Before presenting it as a current forecast product, verify the data source, issue time, update process, and any applicable permissions.

This project is not an official Kenya Meteorological Department (KMD) website or service unless KMD has explicitly approved that status.

## Technology

- HTML
- CSS
- JavaScript
- Inline SVG

## Repository contents

- `index.html` — the interactive forecast explorer.
- `README.md` — project overview and setup instructions.
- `.gitignore` — ignores common local/editor files.

## Contributing

Issues and improvements can be proposed through GitHub Issues or Pull Requests. Please explain changes to forecast data clearly and preserve appropriate source attribution.

## License

No license is included yet. Before adding one, confirm that you have the right to publish and license all code, map geometry, forecast values, and imagery contained in the project.
