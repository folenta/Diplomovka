## Návod na použitie aplikácie 

### Aplikácia pre získavanie podrobných informácií z odtlačkov dlane 

Pre spustenie aplikácie je potrebné spustiť spustiteľný súbor `app.exe`.

Odtlačky z dátovej sady, na ktorej prebehlo testovanie a vyhodnotenie boli dodané Ústavom antropológie Prírodovedeckej fakulty Masarykovej univerzity v Brne a z licenčných dôvodov nie sú súčasťou média. 
Spomínanú dátovú sadu môže okrem Masarykovej univerzity poskytnúť aj výskumná skupina STRaDe.

Anotované dáta vo formáte XML používané v aplikácii aj pri vyhodnotení sú uložené v priečinku `anotatedData`. Po exportovaní údajov o spracovanom odtlačku v aplikácii sú dané údaje ukladané do priečinku `out`. 

### Vyhodnotenie aplikácie

Skript pre vyhodnotenie aplikácie `evaluation.py` porovná anotované dáta z priečinka `anotatedData` so získanými dátami v priečinku `outEvaluation`. Skript pre vyhodnotenie vyžaduje prítomnosť knižnice `numpy`.
Spustenie skriptu je možné jednoduchým príkazom:

```
python evaluation.py
```