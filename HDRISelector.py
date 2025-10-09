import json
import random
import kubric as kb
from typing import Set, Tuple, Optional, List
import numpy as np
import logging

class HDRISelector:
    """
    Una classe avanzata per selezionare HDRI in base a un valore di luminosità continuo [0, 1],
    sfruttando i metadati EV (Exposure Value) e un sistema di pesi per tag e categorie.
    """
    # Mappa dei pesi per tag e categorie. Valori positivi indicano luminosità,
    # valori negativi indicano oscurità.
    LUMINOSITY_WEIGHTS = {
        "sunny": 2.0, "bright": 1.5, "day": 1.0, "clear": 1.0, "sun": 2.0, "midday": 1.5, 
        "blue sky": 1.0, "pure skies": 1.0, "outdoor": 0.5, "field": 0.2, "meadow": 0.2, 
        "lawn": 0.2, "nature": 0.2, "overcast": -0.5, "cloudy": -0.5, "soft": -0.2, "fog": -0.8,
        "night": -2.0, "moon": -1.5, "stars": -1.5, "dark": -1.5, "gloomy": -1.0, "indoor": -1.0, 
        "studio": -1.0, "artificial light": -1.0, "low light": -1.0, "sunrise": 0.0, "sunset": 0.0, 
        "twilight": -0.8, "evening": -0.5, "dawn": -0.2,
    }

    def __init__(self, source: kb.AssetSource, json_path: str):
        """
        Inizializza HDRISelector.
        - source: AssetSource di Kubric, obbligatorio.
        - json_path: Percorso del JSON di Poly Haven, obbligatorio.
        """
        self.source = source
        self.json_path = json_path
        self.available_asset_ids = set(source._assets.keys()) if source and hasattr(source, '_assets') else set()
        
        self.hdri_data = {}
        self._process_json_data()

    def _process_json_data(self):
        """
        Elabora il JSON per calcolare un punteggio di luminosità normalizzato per ogni HDRI.
        """
        try:
            with open(self.json_path, "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logging.error(f"Impossibile leggere o processare il file JSON: {e}")
            return

        for hdri_id, meta in data.items():
            ev_score = float(meta.get("evs", 0))
            tags = set(t.lower() for t in meta.get("tags", []))
            categories = set(c.lower() for c in meta.get("categories", []))
            all_keywords = tags.union(categories)
            
            tag_score = sum(self.LUMINOSITY_WEIGHTS.get(kw, 0) for kw in all_keywords)
            raw_score = (ev_score * 3) + tag_score
            self.hdri_data[hdri_id] = {"raw_score": raw_score}

        all_scores = [d["raw_score"] for d in self.hdri_data.values()]
        
        if not all_scores:
            logging.warning("Nessun punteggio calcolato dal file JSON. La normalizzazione non può essere eseguita.")
            return

        min_score, max_score = min(all_scores), max(all_scores)
        
        for hdri_id in self.hdri_data:
            raw = self.hdri_data[hdri_id]["raw_score"]
            if (max_score - min_score) != 0:
                normalized = (raw - min_score) / (max_score - min_score)
            else:
                normalized = 0.5  # Se tutti gli HDRI hanno lo stesso punteggio, assegna un valore medio
            self.hdri_data[hdri_id]["norm_brightness"] = normalized

    def pick(self, luminosity: float, k: int = 10, rng: Optional[np.random.RandomState] = None) -> str:
        """
        Ritorna un HDRI casuale il cui punteggio di luminosità è più vicino a quello richiesto.
        - luminosity: Valore di luminosità desiderato [0..1].
        - k: Numero dei migliori candidati tra cui scegliere casualmente.
        - rng: Generatore di numeri casuali per la riproducibilità. Se None, usa `random`.
        """
        if not (0.0 <= luminosity <= 1.0):
            raise ValueError("La luminosità deve essere compresa tra 0 e 1.")
        choice_fn = rng.choice if rng else random.choice

        candidates = {
            hdri_id: data for hdri_id, data in self.hdri_data.items()
            if hdri_id in self.available_asset_ids and "norm_brightness" in data
        }
        
        if not candidates:
            logging.warning("Nessun HDRI del JSON trovato nell'AssetSource. Scelta casuale tra tutti gli asset disponibili.")
            return choice_fn(list(self.available_asset_ids))

        sorted_candidates = sorted(
            candidates.items(),
            key=lambda item: abs(item[1]["norm_brightness"] - luminosity)
        )

        top_k_candidates_items = sorted_candidates[:k]
        
        if not top_k_candidates_items:
            return choice_fn(list(candidates.keys()))

        # Estrai solo gli ID dei migliori candidati
        top_k_ids = [item[0] for item in top_k_candidates_items]
        return choice_fn(top_k_ids)

    def get_all_tags_and_categories(self) -> Tuple[Set[str], Set[str]]:
        """
        Restituisce due set contenenti tutti i tag e tutte le categorie uniche
        presenti nel JSON di Poly Haven.
        """
        with open(self.json_path, "r") as f:
            data = json.load(f)

        all_tags, all_categories = set(), set()
        for meta in data.values():
            all_tags.update(t.lower() for t in meta.get("tags", []))
            all_categories.update(c.lower() for c in meta.get("categories", []))

        return all_tags, all_categories