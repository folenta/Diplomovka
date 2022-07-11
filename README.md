Táto diplomová práca sa zaoberá získavaním informácií z odtlačkov dlane. Jej hlavným cieľom je extrahovať z odtlačku dlane informácie o lokálnej orientácii, frekvencii a šírke papilárnych línií, detegovať flekčné ryhy a trirádiá a sledovať priebeh hlavných línií vrátane určenia indexu hlavných línií. Výsledkom tejto práce je aplikácia s grafickým užívateľským rozhraním, ktorá okrem získania informácii z odtlačku dlane tieto informácie graficky zobrazí, štatisticky spracuje a umožní ich export. Testovanie a vyhodnotenie prebehlo na dátovej sade poskytnutej Ústavom antropológie Prírodovedeckej fakulty Masarykovej univerzity v Brne, na ktorej dosiahla detekcia trirádií úspešnosť 84,8 %, úspešnosť určenia ukončenia hlavných línií dosiahla hodnotu 85,38 % a určenie indexu hlavných línií dosahovalo úspešnosť 78 %.

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
