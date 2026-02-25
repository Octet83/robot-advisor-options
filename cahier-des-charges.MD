Agis en tant que Développeur Full-Stack Senior et Analyste Quantitatif expert en options financières (spécialisé dans la Prime de Risque de Volatilité - VRP - et les méthodologies mathématiques type Tastytrade). 

Ton objectif est de coder une application web complète en Python agissant comme un "Robo-Advisor" pour le trading d'options. L'outil doit analyser les données de marché en temps réel et recommander la stratégie mathématiquement optimale à l'instant T, tout en respectant strictement un budget maximal imposé.

Utilise le framework `Streamlit` pour l'interface web (UI) et `yfinance`, `pandas`, `numpy`, et `scipy` pour le backend (récupération des données, calcul du modèle de Black-Scholes pour le Delta et les probabilités).

Voici le cahier des charges strict :

### 1. INPUTS UTILISATEUR (Sidebar Streamlit)
- Un champ texte : `Ticker` (ex: SPY, AAPL, TSLA). Valeur par défaut : SPY. ça doit être un champ de recherche avec auto-complétion.
- Un champ numérique : `Budget Maximum Risqué ($)`. C'est le capital maximum absolu que l'utilisateur est prêt à perdre ou bloquer en marge. L'outil ne doit JAMAIS proposer une stratégie dont le Max Risk dépasse ce montant. Valeur par défaut : 1000.
- Un menu déroulant : `Biais Directionnel` (Haussier, Baissier, Neutre). Valeur par défaut : Neutre.

### 2. MOTEUR DE DONNÉES (Backend)
Dès que l'utilisateur clique sur "Analyser", le script doit récupérer dynamiquement :
- Le prix actuel (Spot) du Ticker.
- Le niveau actuel de l'indice de la peur (Ticker : `^VIX`).
- Calcul de l'IV Rank (Implied Volatility Rank) sur 1 an (252 jours) : Récupère l'historique de la volatilité implicite. Si l'API gratuite ne la fournit pas, calcule la Volatilité Historique (écart-type des rendements) sur 1 an comme proxy. Formule de l'IV Rank : `100 * (IV Actuelle - IV Min 52s) / (IV Max 52s - IV Min 52s)`.
- La chaîne d'options (Options Chain) : Filtre pour trouver la date d'expiration (DTE) la plus proche de **45 jours** (fourchette acceptable : 35 à 60 jours).
- Calcul des Grecques : Implémente une fonction Black-Scholes via `scipy.stats` pour calculer le Delta de chaque option trouvée.

### 3. MATRICE DE DÉCISION ALGORITHMIQUE (Le Cerveau)
L'algorithme doit parser la chaîne d'options (~45 DTE) et sélectionner la stratégie selon ces règles :

- CAS A : Volatilité Élevée (IV Rank > 50 ou VIX > 20) -> VENTE DE PRIME
  - Neutre : **Iron Condor**. (Vendre Call et Put à ~0.16 Delta. Acheter protections).
  - Haussier : **Bull Put Spread**. (Vendre Put à ~0.20 Delta. Acheter protection inférieure).
  - Baissier : **Bear Call Spread**. (Vendre Call à ~0.20 Delta. Acheter protection supérieure).
  - *Sizing :* Largeur du spread * 100 - Prime reçue = Risque Max. Adapte les strikes de protection pour que le Risque Max soit <= Budget Maximum.

- CAS B : Volatilité Faible (IV Rank < 20 et VIX < 15) -> ACHAT DE TEMPS
  - Haussier : **PMCC (Poor Man's Covered Call / Diagonal Spread)**. Achat Call LEAPS (> 200 DTE, Delta > 0.80), Vente Call court (~45 DTE, Delta ~0.30).
  - Neutre : **Calendar Spread**. Achat et Vente à la monnaie sur des expirations différentes (Vente à ~20 DTE, Achat à ~60 DTE).
  - *Sizing :* Débit Net Payé * 100 = Risque Max. Doit être <= Budget Maximum.

- CAS C : Volatilité Moyenne (20 <= IV Rank <= 50) 
  - Proposer des Spreads directionnels classiques ou la stratégie "Cash Secured Put" (The Wheel) si le budget le permet (Budget > Prix de l'action * 100).

### 4. OUTPUTS : LE TABLEAU DE BORD (Interface UI)
Génère une interface visuelle, élégante et financière contenant :
1. CONTEXTE MACRO : Affichage du Spot, du VIX et de l'IV Rank (utilise `st.metric` avec un code couleur : Rouge = Volatilité haute/Danger, Vert = Volatilité basse/Calme).
2. LE VERDICT : Nom de la stratégie recommandée avec 2 phrases expliquant l'avantage mathématique actuel.
3. TICKET D'ORDRE (LEGS) : Un tableau propre affichant les contrats exacts à exécuter. Colonnes : [Action (Buy/Sell)] | [Type (Call/Put)] | [Strike] | [Expiration] | [DTE] | [Prix unitaire].
4. MÉTRIQUES FINANCIÈRES :
   - Crédit net reçu ou Débit net payé en $.
   - **Risque Maximal Exact ($)** (qui prouve que le budget est respecté).
   - Profit Maximal ($).
   - **Probabilité de Profit (PoP %)** : Approximée via le Delta (ex: Vente d'un spread à 0.16 Delta ≈ 84% de chances de réussite).
5. LE PLAN DE VOL (Triggers Mécaniques de Sortie) : Affiche des encarts d'alerte (`st.info` / `st.warning`) avec des consignes strictes :
   - 🎯 TAKE PROFIT : "Placez un ordre limite (GTC) pour racheter la position et encaisser dès que le profit atteint **[Calculer et afficher 50% du Profit Max en $]**."
   - ⏱️ TIME STOP : "Clôturez obligatoirement la position le **[Calculer et afficher la Date d'Expiration exacte - 21 jours]** (à 21 DTE), quels que soient les gains/pertes, pour écraser le risque Gamma."

### 5. CONTRAINTES TECHNIQUES DE CODE
- Fournis le code Python complet, propre, modulaire et commenté dans **un seul fichier `app.py`**.
- Gère les exceptions avec des blocs `try/except` (ex: Ticker invalide, marché fermé rendant les chaînes vides, budget trop faible pour la taille standard d'un contrat de 100 actions).
- Ajoute au début de ta réponse les instructions d'installation (`pip install...`) et la commande pour lancer l'application locale Streamlit.