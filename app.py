import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prédiction", page_icon="📊", layout="centered")

BASE = os.path.dirname(os.path.abspath(__file__))


# configuration des 4 modèles
# l'ordre des champs = l'ordre des colonnes utilisé à l'entraînement
# "enc" = position de l'encodeur (et de la liste des valeurs uniques) de la variable dans les fichiers encoders.joblib / uniques.joblib.

def num(nom, label, **kw):
    return {"nom": nom, "label": label, "type": "num", "kw": kw}


def cat(nom, label, enc):
    return {"nom": nom, "label": label, "type": "cat", "enc": enc}


MODELES = {
    "taxi": {
        "titre": "Prix d'une course de taxi",
        "description": "Estime le prix d'une course selon la distance, la durée, le trafic et la météo.",
        "emoji": "🚕",
        "dossier": "taxi",
        "type": "regression",
        "unite": "$",
        "libelle_resultat": "Prix estimé de la course",
        "colonne_pred": "Prix_Predit",
        "champs": [
            num("Trip_Distance_km", "Distance du trajet (km)", min_value=0.0, value=10.0, step=0.5),
            cat("Time_of_Day", "Moment de la journée", 0),
            cat("Day_of_Week", "Jour de la semaine", 1),
            num("Passenger_Count", "Nombre de passagers", min_value=1, value=1, step=1),
            cat("Traffic_Conditions", "Conditions de circulation", 2),
            cat("Weather", "Météo", 3),
            num("Base_Fare", "Tarif de base", min_value=0.0, value=3.5, step=0.1),
            num("Per_Km_Rate", "Tarif par km", min_value=0.0, value=1.0, step=0.05),
            num("Per_Minute_Rate", "Tarif par minute", min_value=0.0, value=0.3, step=0.05),
            num("Trip_Duration_Minutes", "Durée du trajet (minutes)", min_value=1.0, value=30.0, step=1.0),
        ],
    },
    "voiture_prix": {
        "titre": "Prix de vente d'une voiture d'occasion",
        "description": "Estime le prix de vente à partir du kilométrage, du prix actuel, du carburant et de l'âge.",
        "emoji": "🚗",
        "dossier": "voiture_prix",
        "type": "regression",
        "unite": "$",
        "libelle_resultat": "Prix de vente estimé",
        "colonne_pred": "Prix_Vente_Predit",
        "champs": [
            num("Kms_Driven", "Kilométrage parcouru (km)", min_value=0, value=70000, step=1000),
            num("Present_Price", "Prix actuel (neuf)", min_value=0.0, value=15.0, step=0.5),
            cat("Fuel_Type", "Type de carburant", 0),
            cat("Seller_Type", "Type de vendeur", 1),
            cat("Transmission", "Transmission", 2),
            num("Age", "Âge (années)", min_value=0, value=8, step=1),
        ],
    },
    "voiture_etat": {
        "titre": "Voiture « venant » ou d'occasion",
        "description": "Détermine si une voiture est « Venant » ou « D'occasion » selon sa marque, son année et son prix.",
        "emoji": "🚘",
        "dossier": "voiture_etat",
        "type": "classification",
        "cible_enc": 3,
        "libelle_resultat": "Type de voiture prédit",
        "colonne_pred": "Type_Predit",
        "champs": [
            cat("Marque", "Marque", 0),
            num("Année", "Année", min_value=1990, max_value=2030, value=2015, step=1),
            cat("Boite_Vitesse", "Boîte de vitesse", 1),
            cat("Adresse", "Adresse", 2),
            num("Prix", "Prix", min_value=0, value=5000000, step=100000),
        ],
    },
    "telephone": {
        "titre": "État d'un téléphone",
        "description": "Prédit l'état d'un téléphone (D'occasion, Neuf, Réconditionné ou Venant).",
        "emoji": "📱",
        "dossier": "telephone",
        "type": "classification",
        "cible_enc": 2,
        "libelle_resultat": "État prédit du téléphone",
        "colonne_pred": "Etat_Predit",
        "champs": [
            num("prix", "Prix", min_value=0, value=300000, step=10000),
            cat("adresse", "Adresse", 0),
            cat("marque", "Marque", 1),
            num("dim_ecr", "Dimension de l'écran (pouces)", min_value=0.0, value=6.0, step=0.1, format="%.1f"),
            num("ram", "RAM (Go)", min_value=1, value=8, step=1),
            num("stockage", "Stockage (Go)", min_value=1, value=128, step=1),
        ],
    },
}
ORDRE = ["taxi", "voiture_prix", "voiture_etat", "telephone"]


# chargement et prédiction

@st.cache_resource
def charger(cle):
    dossier = os.path.join(BASE, "models", MODELES[cle]["dossier"])
    return {
        "encoders": joblib.load(os.path.join(dossier, "encoders.joblib")),
        "modele": joblib.load(os.path.join(dossier, "rf_model.joblib")),
        "uniques": joblib.load(os.path.join(dossier, "uniques.joblib")),
        "scaler": joblib.load(os.path.join(dossier, "scaler.joblib")),
    }


def predire(cle, df):
    """df : une ligne par observation, avec les colonnes du modèle (valeurs brutes, sans NaN)."""
    cfg, m = MODELES[cle], charger(cle)
    colonnes = []
    for ch in cfg["champs"]:
        col = df[ch["nom"]]
        if ch["type"] == "cat":
            col = col.astype(str)
            inconnues = sorted(set(col) - set(map(str, m["uniques"][ch["enc"]])))
            if inconnues:
                raise ValueError(
                    f"Colonne « {ch['nom']} » : valeur(s) inconnue(s) {inconnues[:5]}. "
                    "Utilisez les mêmes écritures que dans le jeu de données."
                )
            col = m["encoders"][ch["enc"]].transform(col)
        else:
            col = pd.to_numeric(col, errors="coerce")
            if col.isna().any():
                raise ValueError(f"Colonne « {ch['nom']} » : valeurs manquantes ou non numériques.")
        colonnes.append(np.asarray(col, dtype=float))

    x = m["scaler"].transform(np.column_stack(colonnes))
    y = m["modele"].predict(x)
    if cfg["type"] == "classification":
        return m["encoders"][cfg["cible_enc"]].inverse_transform(y.astype(int))
    return np.round(y, 2)


def lignes_completes(cle, df):
    """True pour les lignes sans valeur manquante (ou non numérique) dans les colonnes du modèle."""
    ok = pd.Series(True, index=df.index)
    for ch in MODELES[cle]["champs"]:
        col = df[ch["nom"]]
        if ch["type"] == "cat":
            ok &= col.notna() & (col.astype(str).str.strip() != "")
        else:
            ok &= pd.to_numeric(col, errors="coerce").notna()
    return ok


def predire_lot(cle, df):
    """Prédit les lignes complètes ; les autres reçoivent une valeur vide."""
    ok = lignes_completes(cle, df)
    if not ok.any():
        raise ValueError("Aucune ligne complète : chaque ligne a au moins une valeur manquante.")
    resultat = pd.Series([None] * len(df), index=df.index, dtype="object")
    resultat[ok] = list(predire(cle, df[ok]))
    return resultat, ok


def lire_csv(fichier):
    try:
        return pd.read_csv(fichier, sep=None, engine="python")
    except UnicodeDecodeError:
        fichier.seek(0)
        return pd.read_csv(fichier, sep=None, engine="python", encoding="latin-1")


# style

GLISSE = "".join(
    f"@keyframes glisse_{sens}_{p} {{from {{opacity: 0; "
    f"transform: translateX({'' if sens == 'av' else '-'}90px) scale(.94);}}}}\n"
    for sens in ("av", "ar") for p in (0, 1)
)

CSS = """
<style>
:root {--encre: #1b2a41; --marine: #1f3a5f; --marine-fonce: #14263f; --trait: #dfe5ef;}

.titre-page {background: #fff; color: var(--marine); border-radius: 50px; padding: .55rem 1rem;
             text-align: center; font-size: 2.3rem; letter-spacing: .02em; max-width: 430px;
             margin: 0 auto 2rem;
             box-shadow: 0 5px 0 #d5dce8, 0 14px 28px rgba(20,38,63,.14);}

/* carte du carrousel */
[class*="st-key-carte_"] {
  background: linear-gradient(160deg, #ffffff 0%, #eef2f9 100%);
  border: 1px solid var(--trait); border-radius: 24px; padding: 1.3rem 1.3rem 1.1rem;
  box-shadow: inset 0 1px 0 #fff,
              0 2px 4px rgba(20,38,63,.06),
              0 10px 20px rgba(20,38,63,.10),
              0 28px 50px rgba(20,38,63,.12);
  transition: transform .35s cubic-bezier(.22,.8,.3,1), box-shadow .35s;
}
[class*="st-key-carte_"]:hover {
  transform: translateY(-6px) perspective(900px) rotateX(1.5deg);
  box-shadow: inset 0 1px 0 #fff,
              0 4px 8px rgba(20,38,63,.08),
              0 16px 28px rgba(20,38,63,.14),
              0 40px 70px rgba(20,38,63,.16);
}
.carte-image {height: 170px; border-radius: 16px; display: flex; align-items: center;
              justify-content: center; font-size: 3.2rem;
              background: radial-gradient(circle at 30% 25%, #ffffff 0%, #dbe5f3 100%);
              box-shadow: inset 0 2px 6px rgba(20,38,63,.12);}
.carte-image span {filter: drop-shadow(0 8px 8px rgba(20,38,63,.28));}
[data-testid="stImage"] img {height: 170px; object-fit: cover; border-radius: 16px;
                             box-shadow: 0 8px 18px rgba(20,38,63,.22);}
.carte-titre {font-size: 1.35rem; font-weight: bold; text-align: center; margin-top: 1rem;
              color: var(--marine);}
.carte-desc {text-align: center; color: #4a5a70; margin: .25rem 0 1rem;}

/* points du carrousel */
.points {display: flex; justify-content: center; align-items: center; gap: .55rem; padding-top: .7rem;}
.point {width: 10px; height: 10px; border-radius: 50%; background: #c3cddc; transition: all .3s;}
.point.actif {width: 28px; border-radius: 10px; background: var(--marine);}

/* boutons en relief */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] button {
  width: 100%; border: none; border-radius: 999px; padding: .5rem 1rem;
  background: linear-gradient(180deg, #2c4f7c 0%, var(--marine) 100%);
  box-shadow: 0 4px 0 var(--marine-fonce), 0 8px 14px rgba(20,38,63,.25);
  transition: transform .12s, box-shadow .12s, filter .2s;}
.stButton > button, .stButton > button p, .stDownloadButton > button, .stDownloadButton > button p,
[data-testid="stFormSubmitButton"] button, [data-testid="stFormSubmitButton"] button p {color: #fff;}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] button:hover {filter: brightness(1.12); color: #fff;}
.stButton > button:active, .stDownloadButton > button:active,
[data-testid="stFormSubmitButton"] button:active {
  transform: translateY(3px); box-shadow: 0 1px 0 var(--marine-fonce), 0 3px 6px rgba(20,38,63,.25);}

/* sections */
.pilule {background: #fff; color: var(--marine); border-radius: 40px; padding: .6rem 1.5rem;
         text-align: center; font-weight: bold; font-size: 1.4rem; max-width: 340px;
         margin: 2.4rem auto 1.4rem;
         box-shadow: 0 5px 0 #d5dce8, 0 12px 24px rgba(20,38,63,.14);}
div[data-testid="stForm"] {
  border: 1px solid #c9d3e3; border-radius: 14px; padding: 1.5rem 1.5rem 1rem 4.2rem;
  box-shadow: 0 10px 28px rgba(20,38,63,.12);
  background-color: #fff;
  background-image:
    linear-gradient(90deg, transparent 3.2rem, #e57373 3.2rem, #e57373 3.3rem, transparent 3.3rem),
    repeating-linear-gradient(180deg, transparent 0, transparent 31px, #d9e8f8 31px, #d9e8f8 32px);
}
.resultat {display: flex; align-items: center; justify-content: center; gap: 1.5rem;
           background: #fff; border-radius: 70px; padding: 1.3rem 2rem; font-size: 1.3rem;
           margin-top: 1.8rem; color: var(--encre);
           box-shadow: 0 6px 0 #d5dce8, 0 16px 32px rgba(20,38,63,.16);}
.resultat .icone {font-size: 3rem; filter: drop-shadow(0 6px 6px rgba(20,38,63,.25));}
.tableau-noir {background: #4a5a55; border: 14px solid #e8c39e; border-radius: 10px;
               padding: 14px; overflow-x: auto; box-shadow: 0 14px 30px rgba(20,38,63,.22);}
.tableau-noir table {border-collapse: collapse; width: 100%; color: #fff;}
.tableau-noir th, .tableau-noir td {border: 2px solid #111; padding: .45rem .6rem; text-align: center;}
.block-container {padding-top: 3rem; max-width: 620px;}
</style>
"""


def pilule(texte):
    st.markdown(f'<div class="pilule">{texte}</div>', unsafe_allow_html=True)


# interface

def aller(delta):
    st.session_state.idx = (st.session_state.idx + delta) % len(ORDRE)
    st.session_state.sens = delta
    st.session_state.nav += 1
    # on ferme la section de la carte précédente
    st.session_state.cle = None
    st.session_state.mode = None
    st.session_state.res_simple = None


def ouvrir(cle, mode):
    st.session_state.cle = cle
    st.session_state.mode = mode
    st.session_state.res_simple = None


def carrousel():
    idx = st.session_state.idx
    cle = ORDRE[idx]
    cfg = MODELES[cle]

    # le nom de l'animation change à chaque navigation, ce qui la relance
    nom = f"glisse_{'av' if st.session_state.sens > 0 else 'ar'}_{st.session_state.nav % 2}"
    st.markdown(
        f"<style>[class*='st-key-carte_'] {{animation: {nom} .55s cubic-bezier(.22,.8,.3,1) backwards;}}</style>",
        unsafe_allow_html=True,
    )

    with st.container(key=f"carte_{cle}"):
        image = next(
            (p for e in ("jpg", "jpeg", "png", "webp")
             if os.path.exists(p := os.path.join(BASE, "images", f"{cle}.{e}"))),
            None,
        )
        if image:
            st.image(image)
        else:
            st.markdown(f'<div class="carte-image"><span>{cfg["emoji"]}</span></div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="carte-titre">{cfg["titre"]}</div>'
            f'<div class="carte-desc">{cfg["description"]}</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.button("Saisie simple", key=f"simple_{cle}", on_click=ouvrir, args=(cle, "simple"))
        c2.button("Saisie multiple", key=f"multiple_{cle}", on_click=ouvrir, args=(cle, "multiple"))

    points = "".join(
        f'<span class="point{" actif" if i == idx else ""}"></span>' for i in range(len(ORDRE))
    )
    g, centre, d = st.columns([1, 2, 1])
    g.button("◀", key="precedent", on_click=aller, args=(-1,))
    centre.markdown(f'<div class="points">{points}</div>', unsafe_allow_html=True)
    d.button("▶", key="suivant", on_click=aller, args=(1,))


def section_simple(cle):
    cfg = MODELES[cle]
    pilule("Saisie simple ✏️")
    st.caption(cfg["titre"])
    m = charger(cle)

    with st.form(f"formulaire_{cle}"):
        valeurs = {}
        for ch in cfg["champs"]:
            cle_w = f"{cle}_{ch['nom']}"
            if ch["type"] == "num":
                valeurs[ch["nom"]] = st.number_input(ch["label"], key=cle_w, **ch["kw"])
            else:
                valeurs[ch["nom"]] = st.selectbox(ch["label"], m["uniques"][ch["enc"]], key=cle_w)
        envoye = st.form_submit_button("Prédire")

    if envoye:
        try:
            y = predire(cle, pd.DataFrame([valeurs]))[0]
            texte = f"{y:.2f} {cfg['unite']}" if cfg["type"] == "regression" else str(y)
            st.session_state.res_simple = (cle, texte)
        except ValueError as e:
            st.session_state.res_simple = None
            st.error(str(e))

    res = st.session_state.get("res_simple")
    if res and res[0] == cle:
        st.markdown(
            f'<div class="resultat"><div class="icone">🖥️</div>'
            f'<div>{cfg["libelle_resultat"]} : <b>{res[1]}</b></div></div>',
            unsafe_allow_html=True,
        )


def section_multiple(cle):
    cfg = MODELES[cle]
    noms = [ch["nom"] for ch in cfg["champs"]]
    pilule("Saisie multiple 📄")
    st.caption(cfg["titre"])
    st.markdown("Colonnes attendues dans le fichier csv : " + ", ".join(f"`{n}`" for n in noms))
    st.download_button(
        "Télécharger un modèle de fichier csv",
        pd.DataFrame(columns=noms).to_csv(index=False).encode("utf-8"),
        file_name=f"modele_{cle}.csv",
        mime="text/csv",
        key=f"modele_{cle}",
    )
    fichier = st.file_uploader("Importer un fichier csv", type="csv", key=f"import_{cle}")
    if fichier is None:
        return

    with st.spinner("En attente de l'affichage de la prédiction…"):
        try:
            df = lire_csv(fichier)
            manquantes = [n for n in noms if n not in df.columns]
            if manquantes:
                st.error(f"Colonnes manquantes dans le fichier : {manquantes}")
                return
            df[cfg["colonne_pred"]], ok = predire_lot(cle, df)
        except ValueError as e:
            st.error(str(e))
            return

    ignorees = int((~ok).sum())
    if ignorees:
        lignes = ", ".join(str(i + 2) for i in df.index[~ok][:10])  # la ligne 1 est l'en-tête
        suite = "…" if ignorees > 10 else ""
        st.warning(
            f"{ignorees} ligne(s) ignorée(s) car incomplète(s) (valeur manquante ou non numérique). "
            f"Lignes concernées dans le fichier : {lignes}{suite}"
        )
    st.success(f"{int(ok.sum())} ligne(s) prédite(s) sur {len(df)}.")

    apercu = df.head(50).to_html(index=False, border=0, na_rep="").replace("\n", "")
    st.markdown(f'<div class="tableau-noir">{apercu}</div>', unsafe_allow_html=True)
    st.caption(f"Aperçu des {min(50, len(df))} premières lignes sur {len(df)}.")
    st.download_button(
        "Télécharger les prédictions (csv)",
        df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"predictions_{cle}.csv",
        mime="text/csv",
        key=f"telecharger_{cle}",
    )


def main():
    st.session_state.setdefault("idx", 0)
    st.session_state.setdefault("sens", 1)
    st.session_state.setdefault("nav", 0)
    st.session_state.setdefault("cle", None)
    st.session_state.setdefault("mode", None)
    st.session_state.setdefault("res_simple", None)

    st.markdown(CSS + f"<style>{GLISSE}</style>", unsafe_allow_html=True)
    st.markdown('<div class="titre-page">Prédiction</div>', unsafe_allow_html=True)
    carrousel()

    cle, mode = st.session_state.cle, st.session_state.mode
    if cle is None:
        return
    try:
        charger(cle)
    except FileNotFoundError:
        st.error(f"Fichiers du modèle introuvables dans models/{MODELES[cle]['dossier']}/")
        return
    (section_simple if mode == "simple" else section_multiple)(cle)


if __name__ == "__main__":
    main()