from playwright.sync_api import sync_playwright
import pandas as pd

with sync_playwright() as p:
    browser = p.chromium.launch(headless=False)
    page = browser.new_page()
    page.goto("https://www.investing.com/economic-calendar/", timeout=60000, wait_until="domcontentloaded")
    page.wait_for_selector("tr.js-event-item", timeout=15000)

    input("Ajuste todos os filtros na página do navegador, depois pressione ENTER para capturar a tabela...")

    rows = page.query_selector_all("tr.js-event-item")
    eventos = []
    for row in rows:
        data_attr = row.get_attribute("data-event-datetime")
        data = data_attr[:10] if data_attr else ""
        cols = row.query_selector_all("td")
        if len(cols) > 5:
            evento = cols[3].inner_text().strip()
            atual = cols[4].inner_text().strip()
            previsao = cols[5].inner_text().strip()
            anterior = cols[6].inner_text().strip()
            eventos.append({
                'data': data,
                'evento': evento,
                'atual': atual,
                'previsao': previsao,
                'anterior': anterior
            })

    browser.close()

df = pd.DataFrame(eventos)
df.to_csv('economic_calendar_playwright.csv', index=False)
print(df.head())
