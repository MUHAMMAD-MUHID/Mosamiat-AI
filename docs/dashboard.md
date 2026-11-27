# Islamabad weather dashboard

Run these commands from the project root. The dashboard reads the same Hopsworks project as the hourly workflow. It uses the existing `HOPSWORKS_HOST`, `HOPSWORKS_PROJECT` and `HOPSWORKS_API_KEY` environment settings or local `.env` file.

```powershell
& ./.tools/bootstrap/bin/uv.exe sync --locked --inexact --extra dashboard
& ./.venv/Scripts/python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

Open the terminal's local URL. The page checks for updates every five minutes while open. Under **Desk settings**, refresh the forecast or select temperature and wind units. The issue time describes when predictions were made. The verification time under **Forecast sources and credits** records when forecast readback succeeded. All displayed forecast times use Pakistan time.

- **Overview** shows the temperature for the labeled forecast hour, the low and high across the 24-hour window, a summary of air quality, rain and wind, and the highest-priority active or upcoming alert. Select any of the eight forecast variables to see its trend. When the entire window has elapsed, the overview labels the forecast as ended and displays the last forecast hour.
- **Hourly explorer** plots the first 6, 12 or 24 hours and exports that window as CSV in the selected units. The comparison below the table lets you pick any two hours in the full forecast and inspect all eight variables together.
- **Alerts** filters published threshold crossings by status and severity. Switch the audience for specific guidance, or open the evidence for the peak value, threshold and original units.
- **Bulletin** displays the published report with its original issue time and units.

The dashboard reads the scheduler's checksum-addressed ledger in Hopsworks and selects the latest verified forecast by issue time. It checks coverage and units before displaying the eight fields for all 24 forecast hours. Rain probability is stored as a fraction and displayed as a percentage. If its bulletin is still pending, the dashboard computes alerts from the verified forecast and keeps the latest verified report. Drift and retraining have separate statuses.

A verified copy is cached under the ignored `data/dashboard/` directory. If Hopsworks is unavailable, the page labels the cached source or uses the local verified ledger. An expired forecast is marked on the page.

The Islamabad photograph is illustrative. The resized WebP is licensed under the same terms as the original. [Photograph by Imrankhakwani on Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Faisal_Mosque_at_dusk_seen_from_Margalla_Hills.jpg) · [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
