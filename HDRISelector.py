import json
import random
import kubric as kb  # Assicurati di avere Kubric importato
from typing import Set, Tuple, Optional

class HDRISelector:
    # Tag e categorie ampiamente definiti per ciascuna luminosità
    LUMINOSITY_MAP = {
        "bright": {
            "tags": {"sunny", "bright", "day", "clear", "blue_sky", "sun", "green", "open air",
                     "pure skies", "midday", "calm", "coastal", "beach", "field", "meadow",
                     "garden", "vine", "vineyard", "lawn", "forest", "countryside"},
            "categories": {"midday","natural light","outdoor","pure skies","skies","nature"}
        },
        "medium": {
            "tags": {"sunrise", "sunset", "twilight", "evening", "dawn", "dappled", "fog",
                     "partly cloudy", "cloud", "overcast", "soft", "warm", "orange", "red", "yellow"},
            "categories": {"morning-afternoon","sunrise-sunset","partly cloudy","overcast","medium contrast"}
        },
        "dark": {
            "tags": {"night", "moon", "stars", "dark", "gloomy", "shadow", "indoor", "studio",
                     "artificial light", "low_light", "industrial", "storm", "winter", "twilight"},
            "categories": {"night","studio","artificial light","indoor","overcast","low contrast","high contrast"}
        }
    }

    def __init__(self, source: Optional[kb.AssetSource] = None, json_path: Optional[str] = None):
        """
        Inizializza HDRISelector:
        - source: AssetSource di Kubric (mantiene la funzionalità di pick con AssetSource)
        - json_path: percorso del JSON scaricato da Poly Haven
        """
        if source is None and json_path is None:
            raise ValueError("Devi fornire almeno un source o un json_path")

        self.source = source
        self.json_path = json_path
        self.hdri_assets = source._assets if source is not None else None

        # Raggruppamento basato sul JSON
        self.grouped = {"bright": [], "medium": [], "dark": []}
        if self.json_path is not None:
            self._group_assets_from_json()

    def _group_assets_from_json(self):
        """Raggruppa gli HDRI in bright/medium/dark usando solo il JSON"""
        with open(self.json_path, "r") as f:
            data = json.load(f)

        for hdri_id, meta in data.items():
            tags = set(t.lower() for t in meta.get("tags", []))
            categories = set(c.lower() for c in meta.get("categories", []))

            # Calcola punteggio per ciascun gruppo
            scores = {}
            for group, criteria in self.LUMINOSITY_MAP.items():
                score = len(tags & criteria["tags"]) + len(categories & criteria["categories"])
                scores[group] = score

            best_group = max(scores, key=scores.get)
            if scores[best_group] == 0:
                best_group = "bright"  # fallback

            self.grouped[best_group].append(hdri_id)

    def pick(self, luminosity: float) -> str:
        """Ritorna un hdri casuale coerente con la luminosità [0..1]"""
        if self.source is None:
            raise ValueError("pick() richiede un AssetSource fornito all'inizializzazione")

        if luminosity >= 0.75:
            group = "bright"
        elif luminosity >= 0.5:
            group = "medium"
        else:
            group = "dark"

        # Se l'HDRI non è presente nel JSON (non raggruppato), fallback casuale su tutti gli asset
        candidates = [hdri for hdri in self.grouped[group] if hdri in self.hdri_assets]
        if not candidates:
            candidates = list(self.hdri_assets.keys())
            print(f"[WARN] Nessun HDRI nel gruppo '{group}' disponibile nel source, pesco casuale da tutti.")

        return random.choice(candidates)

    def get_all_tags_and_categories(self) -> Tuple[Set[str], Set[str]]:
        """
        Restituisce due set contenenti tutti i tag e tutte le categorie uniche
        presenti nel JSON scaricato da Poly Haven
        """
        if self.json_path is None:
            raise ValueError("get_all_tags_and_categories richiede il percorso json_path")

        with open(self.json_path, "r") as f:
            data = json.load(f)

        all_tags = set()
        all_categories = set()

        for meta in data.values():
            all_tags.update(t.lower() for t in meta.get("tags", []))
            all_categories.update(c.lower() for c in meta.get("categories", []))

        return all_tags, all_categories
