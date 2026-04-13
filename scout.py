import math
import random
import sys
import threading
import numpy as np
from scipy.optimize import nnls
import tbaapiv3client
from collections import defaultdict

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QComboBox, QPushButton, QFrame)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl

# ─── Phrase banks ─────────────────────────────────────────────────────────────
AUTO_WIN_PHRASES = [
    "{name} is set to win the auto and take the early initiative.",
    "Auto control looks likely to go to {name}, giving them a strong start.",
    "{name} appears primed to claim the auto phase and dictate early tempo.",
]
AUTO_LOSE_PHRASES = [
    "{name} may trail through auto and will need to claw back later.",
    "The early auto phase looks like it goes to the opponent, so {name} must recover quickly.",
    "{name} is forecast to miss out on auto control and must rely on later shifts.",
]
ACTIVE_SHIFT_PHRASES = [
    "They are slated for active shifts {active_shifts}, while the opposition is pushed into {inactive_shifts}.",
    "This plan puts their strongest rotations in shifts {active_shifts}, leaving {inactive_shifts} to the opponent.",
]
ACTIVE_STRENGTH_PHRASES = [
    "Active shift strength should supply {predicted_active_fuel:.2f} points.",
    "They should get {predicted_active_fuel:.2f} from active shifts.",
]
TSP_EFR_PHRASES = [
    "Transition and endgame combine for {total:.2f}, split {tsp:.2f} / {efr:.2f}.",
    "Their TSP plus EFR total {total:.2f}, with {tsp:.2f} from transition and {efr:.2f} from endgame.",
]
OUTCOME_PHRASES = [
    "{winner} is expected to win {label}, roughly {winner_performance:.1f} to {loser_performance:.1f}.",
    "Forecast gives {winner} the edge {label}, at about {winner_performance:.1f} versus {loser_performance:.1f}.",
]
STRATEGIC_EDGE_RED_PHRASES = [
    "Red holds the strategic edge in TSS - their selection of plays and rhythm is stronger.",
]
STRATEGIC_EDGE_BLUE_PHRASES = [
    "Blue holds the strategic edge in TSS - their selection of plays and rhythm is stronger.",
]
STRATEGIC_EDGE_TIE_PHRASES = [
    "Strategically, both alliances are matched closely in TSS.",
    "TSS shows the two sides are very even, making this a chess match.",
]

# ─── TBA API ──────────────────────────────────────────────────────────────────
configuration = tbaapiv3client.Configuration(
    host="https://www.thebluealliance.com/api/v3",
    api_key={'X-TBA-Auth-Key': 'z1QRmzpf0rpw4cXVqOYN0hDG7t1XiO60wOBIIaCFg2ksv6WJg9OZd3JBiptNoQxK'}
)
api_client  = tbaapiv3client.ApiClient(configuration)
event_api   = tbaapiv3client.EventApi(api_client)
match_api   = tbaapiv3client.MatchApi(api_client)

# ─── Analytics core ───────────────────────────────────────────────────────────
def safe_float(v):
    try: return float(v)
    except: return 0.0

def clamp_zero(v): return max(0.0, v)
def non_negative(v): return max(0.0, v)

def get_hub_score(match, alliance):
    bd = getattr(match, 'score_breakdown', None)
    if not bd: return {}
    return bd.get(alliance, {}).get('hubScore', {})

def active_shift_counts(hub):
    return [c for c in [safe_float(hub.get(f'shift{i}Count')) for i in range(1, 5)] if c > 0]

def alliance_aggregates(match, alliance):
    hub = get_hub_score(match, alliance)
    sc  = active_shift_counts(hub)
    sd  = abs(sc[0] - sc[1]) if len(sc) >= 2 else (sc[0] if sc else 0.0)
    return {
        'auto_fuel':          safe_float(hub.get('autoCount')),
        'transition_fuel':    safe_float(hub.get('transitionPoints')),
        'active_shift_fuel':  sum(sc),
        'endgame_fuel':       safe_float(hub.get('endgameCount')),
        'tower_points':       safe_float(match.score_breakdown.get(alliance, {}).get('totalTowerPoints', 0)),
        'active_shift_delta': sd,
        'shift_counts':       sc,
    }

def compute_phi(osr_values):
    t = sum(osr_values)
    cap = min(t, 35.0)
    pen = max(0.0, (t - 35.0) / 100.0)
    return clamp_zero(cap * (1.0 - pen))

def compute_hip(a, b):
    d = a + b
    return a / d if d > 0 else 0.0

def solve_nnls(A, b):
    if not A or not b: return []
    a = np.asarray(A, dtype=float)
    bv = np.asarray(b, dtype=float)
    if a.ndim != 2 or bv.ndim != 1 or a.shape[0] != bv.shape[0]: return []
    try: x, _ = nnls(a, bv)
    except: x = np.maximum(np.linalg.lstsq(a, bv, rcond=None)[0], 0.0)
    return x.tolist()

def build_design_matrix(matches, robot_index, metric_key, match_weights):
    M, T = [], []
    for match, al in matches:
        v = alliance_aggregates(match, al)[metric_key]
        if math.isnan(v) or math.isinf(v): continue
        w = match_weights.get(getattr(match, 'key', None), 1.0)
        sqrt_w = math.sqrt(w)
        row = [0.0] * len(robot_index)
        for r in getattr(match.alliances, al).team_keys:
            if r in robot_index: row[robot_index[r]] = 1.0 * sqrt_w
        M.append(row); T.append(v * sqrt_w)
    return M, T

def build_shift_design_matrix(matches, robot_index, si, match_weights):
    M, T = [], []
    for match, al in matches:
        v = safe_float(get_hub_score(match, al).get(f'shift{si}Count'))
        w = match_weights.get(getattr(match, 'key', None), 1.0)
        sqrt_w = math.sqrt(w)
        row = [0.0] * len(robot_index)
        for r in getattr(match.alliances, al).team_keys:
            if r in robot_index: row[robot_index[r]] = 1.0 * sqrt_w
        M.append(row); T.append(v * sqrt_w)
    return M, T

def active_shifts_from_auto_winner(ar, ab):
    return {'red': [2, 4], 'blue': [1, 3]} if ar >= ab else {'red': [1, 3], 'blue': [2, 4]}

def extract_event_match_data(event_key):
    try: all_m = event_api.get_event_matches(event_key=event_key)
    except:
        try: keys = event_api.get_event_matches_keys(event_key=event_key)
        except: keys = []
        all_m = []
        for k in keys:
            try: all_m.append(match_api.get_match(match_key=k))
            except: pass
    return [m for m in all_m if getattr(m, 'score_breakdown', None) and getattr(m, 'alliances', None)]

def build_robot_index(matches):
    robots = set()
    for m in matches:
        for ac in ['blue', 'red']: robots.update(getattr(m.alliances, ac).team_keys)
    return {r: i for i, r in enumerate(sorted(robots))}

def collect_metric_coefficients(matches, robot_index, match_weights):
    alls = [(m, ac) for m in matches for ac in ['blue', 'red']]
    coeffs = {}
    for metric in ['auto_fuel', 'transition_fuel', 'active_shift_fuel', 'endgame_fuel', 'active_shift_delta']:
        M, T = build_design_matrix(alls, robot_index, metric, match_weights)
        coeffs[metric] = solve_nnls(M, T)
    coeffs['shift_osr'] = {}
    for si in range(1, 5):
        M, T = build_shift_design_matrix(alls, robot_index, si, match_weights)
        coeffs['shift_osr'][si] = solve_nnls(M, T)
    return coeffs

def compute_dsr_coefficients(matches, robot_index, osr_coeffs, match_weights):
    M, T = [], []
    for match in matches:
        w = match_weights.get(getattr(match, 'key', None), 1.0)
        sqrt_w = math.sqrt(w)
        for ac in ['blue', 'red']:
            opp = 'red' if ac == 'blue' else 'blue'
            opp_osr = sum(osr_coeffs[robot_index[r]] for r in getattr(match.alliances, opp).team_keys if r in robot_index)
            sm = max(0.0, opp_osr - alliance_aggregates(match, opp)['active_shift_fuel'])
            row = [0.0] * len(robot_index)
            for r in getattr(match.alliances, ac).team_keys:
                if r in robot_index: row[robot_index[r]] = 1.0 * sqrt_w
            M.append(row); T.append(sm * sqrt_w)
    return solve_nnls(M, T)

def compute_tower_average(matches, match_weights):
    totals, counts = defaultdict(float), defaultdict(float)
    for match in matches:
        w = match_weights.get(getattr(match, 'key', None), 1.0)
        for ac in ['blue', 'red']:
            tp = safe_float(getattr(match, 'score_breakdown', {}).get(ac, {}).get('totalTowerPoints', 0))
            for r in getattr(match.alliances, ac).team_keys:
                totals[r] += (tp / 3.0) * w
                counts[r] += w
    return {r: totals[r] / max(counts[r], 0.0001) for r in totals}

def build_robot_dictionary(coeff_list, robot_index):
    return {r: (coeff_list[i] if i < len(coeff_list) else 0.0) for r, i in robot_index.items()}

def compute_match_predictions(red_keys, blue_keys, coefficients, direct_ta):
    def S(k, keys): return sum(coefficients[k].get(r, 0.0) for r in keys)
    
    ar, ab = S('afr', red_keys), S('afr', blue_keys)
    schedule = active_shifts_from_auto_winner(ar, ab)
    
    ss_red  = {k: sum(non_negative(coefficients['shift_osr'][k].get(r, 0.0)) for r in red_keys)  for k in range(1, 5)}
    ss_blue = {k: sum(non_negative(coefficients['shift_osr'][k].get(r, 0.0)) for r in blue_keys) for k in range(1, 5)}
    
    pa_red  = sum(ss_red[k] for k in schedule['red'])
    pa_blue = sum(ss_blue[k] for k in schedule['blue'])
    
    dsr_r = S('dsr', red_keys)
    dsr_b = S('dsr', blue_keys)
    
    fuel_r = max(0.0, S('tsp', red_keys) + pa_red + S('efr', red_keys) - dsr_b)
    fuel_b = max(0.0, S('tsp', blue_keys) + pa_blue + S('efr', blue_keys) - dsr_r)
    
    if ar > ab: 
        fuel_r += 0.0 * ss_red[4]            
        fuel_b += 0.2 * S('efr', blue_keys)  
    elif ab > ar: 
        fuel_b += 0.0 * ss_blue[4]           
        fuel_r += 0.2 * S('efr', red_keys)   
    
    ta_r = sum(direct_ta.get(r, 0.0) for r in red_keys)
    ta_b = sum(direct_ta.get(r, 0.0) for r in blue_keys)
    
    sc_r = ar + fuel_r + ta_r
    sc_b = ab + fuel_b + ta_b
    
    phi_r = compute_phi([coefficients['osr'].get(r, 0.0) for r in red_keys])
    phi_b = compute_phi([coefficients['osr'].get(r, 0.0) for r in blue_keys])
    
    hip_r = compute_hip(ar, ab)
    hip_b = compute_hip(ab, ar)
    
    tss_r = phi_r + dsr_r + ta_r + hip_r * 15.0
    tss_b = phi_b + dsr_b + ta_b + hip_b * 15.0
    
    return {
        'red':  {'auto': ar, 'fuel': fuel_r, 'tower': ta_r, 'score': sc_r, 'phi': phi_r, 'hip': hip_r,
                 'active_shifts': schedule['red'],  'predicted_active_fuel': pa_red,  'shift_scores': ss_red,  'tss': tss_r},
        'blue': {'auto': ab, 'fuel': fuel_b, 'tower': ta_b, 'score': sc_b, 'phi': phi_b, 'hip': hip_b,
                 'active_shifts': schedule['blue'], 'predicted_active_fuel': pa_blue, 'shift_scores': ss_blue, 'tss': tss_b},
        'schedule': schedule,
    }

# ─── Loaders ──────────────────────────────────────────────────────────────────
def format_match_label(comp_level, match_number, set_number=None, key=None):
    level = (comp_level or '').lower()
    lmap = {'qm': 'Qualification', 'qf': 'Quarterfinal', 'sf': 'Semifinal',
            'f': 'Final', 'f1': 'Final', 'final': 'Final', 'ef': 'Octofinal'}
    label = lmap.get(level, (comp_level or 'Match').title())
    if level in ('qf', 'sf', 'ef') and set_number is not None: label = f"{label} {set_number}"
    if match_number is not None:
        label = f"{label} Match {match_number}" if level in ('qf', 'sf', 'ef') else f"{label} {match_number}"
    return label.strip() or (key or 'Match')

def load_events_by_year(year=2026):
    try:
        if hasattr(event_api, 'get_events_by_year'): evs = event_api.get_events_by_year(year=year)
        else: evs = event_api.get_events_by_year_simple(year=year)
    except: evs = []
    return sorted([e for e in evs if getattr(e, 'key', None)], key=lambda e: ((e.week if e.week is not None else 0), getattr(e, 'name', '')))

def load_event_weeks(year=2026):
    wm = {}
    for ev in load_events_by_year(year):
        w = getattr(ev, 'week', None)
        if w is not None: wm[str(w + 1)] = w
    return wm or {'1': 0}

def load_events_for_week(display_week, year=2026):
    wm = load_event_weeks(year)
    aw = wm.get(str(display_week), None)
    if aw is None:
        try: aw = int(display_week) - 1
        except: return []
    return [e for e in load_events_by_year(year) if getattr(e, 'week', None) == aw]

def load_match_labels(team_key, event_key):
    try:
        if hasattr(match_api, 'get_team_event_matches_simple'):
            ms = match_api.get_team_event_matches_simple(team_key=team_key, event_key=event_key)
        else:
            ms = match_api.get_team_event_matches(team_key=team_key, event_key=event_key)
    except: return []
    labels = []
    # Sort matches chronologically
    def get_match_time(m):
        comp_order = {'qm': 1, 'ef': 2, 'qf': 3, 'sf': 4, 'f': 5}
        cl = getattr(m, 'comp_level', 'qm')
        return (comp_order.get(cl, 0), getattr(m, 'set_number', 0), getattr(m, 'match_number', 0))
    ms = sorted([m for m in ms if m is not None], key=get_match_time)

    for m in ms:
        key = getattr(m, 'key', None)
        cl  = getattr(m, 'comp_level', None) or getattr(m, 'compLevel', None)
        mn  = getattr(m, 'match_number', None) or getattr(m, 'matchNumber', None)
        sn  = getattr(m, 'set_number', None)   or getattr(m, 'setNumber', None)
        labels.append((format_match_label(cl, mn, sn, key), key))
    return labels

# ─── HTML Header (Shared Base) ───────────────────────────────────────────────
def get_html_head():
    return """
    <!DOCTYPE html>
    <html class="dark" lang="en">
    <head>
        <meta charset="utf-8"/>
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title>Tactical Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet"/>
        <script>
            tailwind.config = {
                darkMode: "class",
                theme: {
                    extend: {
                        colors: {
                            "tertiary": "#74f5ff",
                            "secondary": "#ffb3ae",
                            "surface-dim": "#131315",
                            "surface-container-low": "#1c1b1e",
                            "surface-container-lowest": "#0e0e10",
                            "secondary-container": "#930014",
                        },
                        fontFamily: {
                            "headline": ["Space Grotesk", "sans-serif"],
                            "body": ["Inter", "sans-serif"],
                            "mono": ["JetBrains Mono", "monospace"]
                        }
                    }
                }
            }
        </script>
        <style>
            .grain-overlay { background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E"); opacity: 0.04; pointer-events: none; }
            .glow-red { text-shadow: 0 0 25px rgba(255, 179, 174, 0.4); }
            .glow-cyan { text-shadow: 0 0 25px rgba(116, 245, 255, 0.4); }
            .table-cell-custom { padding: 12px 8px; border-bottom: 1px solid rgba(255,255,255,0.05); font-family: 'JetBrains Mono', monospace; font-size: 13px; text-align: right; }
            .table-header-custom { padding-bottom: 12px; border-bottom: 1px solid rgba(255,255,255,0.1); font-family: 'Space Grotesk', sans-serif; font-size: 10px; font-weight: 700; color: rgba(255,255,255,0.4); text-transform: uppercase; letter-spacing: 0.1em; text-align: right; }
            ::-webkit-scrollbar { width: 8px; height: 8px; }
            ::-webkit-scrollbar-track { background: #08080A; }
            ::-webkit-scrollbar-thumb { background: #1c1b1e; }
            ::-webkit-scrollbar-thumb:hover { background: #353437; }
        </style>
    </head>
    <body class="bg-[#08080A] text-[#e5e1e4] font-body selection:bg-tertiary selection:text-black min-h-screen">
        <div class="fixed inset-0 grain-overlay z-[100]"></div>
    """

# ─── View Generators ──────────────────────────────────────────────────────────

def build_leaderboard_html(event_selected, event_key):
    event_matches = extract_event_match_data(event_key)
    if not event_matches: return f'<div style="color:#ff6b6b; padding: 40px; font-family:sans-serif;">No match data available for {event_selected}.</div>'

    def get_match_time(m):
        comp_order = {'qm': 1, 'ef': 2, 'qf': 3, 'sf': 4, 'f': 5}
        cl = getattr(m, 'comp_level', 'qm')
        return (getattr(m, 'actual_time', None) or getattr(m, 'time', 0) or 0, comp_order.get(cl, 0))
        
    sorted_matches = sorted(event_matches, key=get_match_time)
    N = len(sorted_matches)
    
    MIN_WEIGHT = 0.25
    match_weights = { getattr(m, 'key', None): MIN_WEIGHT + (1.0 - MIN_WEIGHT) * (i / max(1, N - 1)) for i, m in enumerate(sorted_matches) }
    
    robot_index = build_robot_index(event_matches)
    cbm = collect_metric_coefficients(event_matches, robot_index, match_weights)
    dsr_c = compute_dsr_coefficients(event_matches, robot_index, cbm['active_shift_fuel'], match_weights)
    ta = compute_tower_average(event_matches, match_weights)

    C = {
        'afr': build_robot_dictionary(cbm['auto_fuel'], robot_index),
        'osr': build_robot_dictionary(cbm['active_shift_fuel'], robot_index),
        'dsr': build_robot_dictionary(dsr_c, robot_index),
        'efr': build_robot_dictionary(cbm['endgame_fuel'], robot_index),
        'ta':  ta,
    }

    teams_data = []
    for r in robot_index.keys():
        impact = C['afr'].get(r,0) + C['osr'].get(r,0) + C['dsr'].get(r,0) + C['efr'].get(r,0) + C['ta'].get(r,0)
        teams_data.append({
            'key': r,
            'impact': impact,
            'afr': C['afr'].get(r,0),
            'osr': C['osr'].get(r,0),
            'dsr': C['dsr'].get(r,0),
            'efr': C['efr'].get(r,0)
        })
    teams_data.sort(key=lambda x: x['impact'], reverse=True)

    rows = ""
    for i, td in enumerate(teams_data):
        rows += f'''
        <tr onclick="window.location.href='scout://team/{td['key']}'" class="cursor-pointer hover:bg-white/10 transition-colors">
            <td class="table-cell-custom text-left pl-4 font-bold text-tertiary">#{i+1}</td>
            <td class="table-cell-custom text-left text-white/90 font-bold">{td['key']}</td>
            <td class="table-cell-custom text-secondary glow-red font-bold">{td['impact']:.1f}</td>
            <td class="table-cell-custom text-white/70">{td['afr']:.1f}</td>
            <td class="table-cell-custom text-white/70">{td['osr']:.1f}</td>
            <td class="table-cell-custom text-white/70">{td['dsr']:.1f}</td>
            <td class="table-cell-custom text-white/70">{td['efr']:.1f}</td>
        </tr>
        '''

    html_body = f"""
    <main class="max-w-4xl mx-auto pt-10 pb-20 px-6 relative z-10">
        <header class="flex justify-between items-center border-b border-white/10 pb-4 mb-10">
            <div class="flex items-center gap-4">
                <h1 class="font-headline font-bold text-2xl text-white tracking-widest">EVENT LEADERBOARD</h1>
            </div>
            <div class="text-right">
                <span class="font-mono text-[10px] text-tertiary tracking-[0.2em] uppercase block">Scout Analytics</span>
                <span class="font-headline text-xs text-white/40 tracking-widest uppercase">{event_selected}</span>
            </div>
        </header>

        <section class="bg-[#0e0e10] p-1 border border-white/5 overflow-x-auto">
            <table class="w-full whitespace-nowrap">
                <thead>
                    <tr>
                        <th class="table-header-custom text-left pl-4">Rank</th>
                        <th class="table-header-custom text-left">Team</th>
                        <th class="table-header-custom">Total Impact</th>
                        <th class="table-header-custom">AFR</th>
                        <th class="table-header-custom">OSR</th>
                        <th class="table-header-custom">DSR</th>
                        <th class="table-header-custom">EFR</th>
                    </tr>
                </thead>
                <tbody>
                    {rows}
                </tbody>
            </table>
        </section>
    </main>
    </body>
    </html>
    """
    return get_html_head() + html_body

def build_schedule_html(event_selected, event_key, team_key):
    matches = load_match_labels(team_key, event_key)
    
    rows = ""
    for lbl, m_key in matches:
        rows += f'''
        <div onclick="window.location.href='scout://match/{m_key}'" class="bg-[#1c1b1e]/60 p-4 border-l-2 border-tertiary/50 mb-2 cursor-pointer hover:bg-tertiary/20 transition-colors flex justify-between items-center">
            <span class="font-headline font-bold text-white tracking-widest">{lbl}</span>
            <span class="font-mono text-xs text-white/40">VIEW MATCH &rarr;</span>
        </div>
        '''

    html_body = f"""
    <main class="max-w-2xl mx-auto pt-10 pb-20 px-6 relative z-10">
        <div onclick="window.location.href='scout://leaderboard'" class="cursor-pointer text-secondary mb-6 font-mono text-xs hover:text-white transition-colors flex items-center gap-2">
            <span>&larr;</span> BACK TO LEADERBOARD
        </div>

        <header class="border-b border-white/10 pb-4 mb-10">
            <h1 class="font-headline font-bold text-3xl text-tertiary glow-cyan tracking-widest">{team_key.upper()}</h1>
            <span class="font-headline text-xs text-white/40 tracking-widest uppercase mt-2 block">{event_selected} SCHEDULE</span>
        </header>

        <section>
            {rows}
        </section>
    </main>
    </body>
    </html>
    """
    return get_html_head() + html_body

# Change the parameters to accept BOTH event_label and event_key
def build_html_report(event_label, event_key, team_selected, match_selected):
    try: selected_match = match_api.get_match(match_key=match_selected)
    except Exception as exc: return f'<div style="color:#ff6b6b; padding: 40px; font-family:sans-serif;">Error: {exc}</div>'

    # USE THE EVENT_KEY HERE FOR THE API
    event_matches = extract_event_match_data(event_key)
    if not event_matches: return '<div style="color:#ff6b6b; padding: 40px; font-family:sans-serif;">No match data available.</div>'

    def get_match_time(m):
# ... (Keep the rest of the function the same until the HTML section) ...
        comp_order = {'qm': 1, 'ef': 2, 'qf': 3, 'sf': 4, 'f': 5}
        cl = getattr(m, 'comp_level', 'qm')
        return (getattr(m, 'actual_time', None) or getattr(m, 'time', 0) or 0, comp_order.get(cl, 0))
        
    sorted_matches = sorted(event_matches, key=get_match_time)
    N = len(sorted_matches)
    
    MIN_WEIGHT = 0.25
    match_weights = { getattr(m, 'key', None): MIN_WEIGHT + (1.0 - MIN_WEIGHT) * (i / max(1, N - 1)) for i, m in enumerate(sorted_matches) }
    
    robot_index = build_robot_index(event_matches)
    cbm = collect_metric_coefficients(event_matches, robot_index, match_weights)
    dsr_c = compute_dsr_coefficients(event_matches, robot_index, cbm['active_shift_fuel'], match_weights)
    ta = compute_tower_average(event_matches, match_weights)

    C = {
        'afr': build_robot_dictionary(cbm['auto_fuel'], robot_index),
        'tsp': build_robot_dictionary(cbm['transition_fuel'], robot_index),
        'osr': build_robot_dictionary(cbm['active_shift_fuel'], robot_index),
        'efr': build_robot_dictionary(cbm['endgame_fuel'], robot_index),
        'svc': build_robot_dictionary(cbm['active_shift_delta'], robot_index),
        'dsr': build_robot_dictionary(dsr_c, robot_index),
        'ta':  ta,
        'shift_osr': {si: build_robot_dictionary(cbm['shift_osr'][si], robot_index) for si in range(1, 5)},
    }

    blue_keys = getattr(selected_match.alliances, 'blue').team_keys
    red_keys  = getattr(selected_match.alliances, 'red').team_keys
    P = compute_match_predictions(red_keys, blue_keys, C, ta)
    R, B = P['red'], P['blue']

    r_perf = R['score']
    b_perf = B['score']
    winner = 'Red' if r_perf >= b_perf else 'Blue'
    margin = abs(r_perf - b_perf)
    
    w_color_class = "text-secondary glow-red" if winner == 'Red' else "text-tertiary glow-cyan"

    def cl(m): return 'narrowly' if m < 25 else ('comfortably' if m < 50 else 'decisively')
    outcome = random.choice(OUTCOME_PHRASES).format(winner=winner, label=cl(margin), winner_performance=r_perf if winner=='Red' else b_perf, loser_performance=b_perf if winner=='Red' else r_perf)
    strategic = random.choice(STRATEGIC_EDGE_RED_PHRASES if R['tss'] > B['tss'] else (STRATEGIC_EDGE_BLUE_PHRASES if B['tss'] > R['tss'] else STRATEGIC_EDGE_TIE_PHRASES))

    def shift_bars(data, color_theme):
        mx = max(data['shift_scores'].values() or [1])
        base_bg = "bg-secondary/10 border-secondary/30" if color_theme == 'red' else "bg-tertiary/10 border-tertiary/30"
        fill_bg = "bg-secondary/60 border-secondary glow-red" if color_theme == 'red' else "bg-tertiary/60 border-tertiary glow-cyan"
        bars = ""
        for k in range(1, 5):
            v = data['shift_scores'][k]
            pct = max(10, (v / max(mx, 0.001)) * 100)
            is_active = k in data['active_shifts']
            css_class = fill_bg if is_active else base_bg
            bars += f"""
            <div class="flex flex-col justify-end h-24 flex-1 group">
                <div class="text-center font-mono text-[9px] text-white/30 mb-2 group-hover:text-white transition-colors">S{k} <br/> {v:.0f}</div>
                <div class="w-full {css_class} border-t-2" style="height: {pct}%;"></div>
            </div>
            """
        return f'<div class="flex items-end gap-2 h-full w-full">{bars}</div>'

    def robot_rows(keys, color_hex):
        out = ''
        for r in keys:
            osr = C['osr'].get(r, 0.0); svc = C['svc'].get(r, 0.0)
            rs = osr * (1.0 - svc / (osr + 1.0)) if (osr + 1.0) != 0.0 else 0.0
            shifts_html = ''.join(f'<td class="table-cell-custom text-white/70">{clamp_zero(C["shift_osr"][si].get(r, 0.0)):.1f}</td>' for si in range(1, 5))
            out += f'''
                <tr class="hover:bg-white/5 transition-colors">
                    <td class="table-cell-custom text-left" style="color: {color_hex}; font-weight: bold; letter-spacing: 1px;">{r}</td>
                    <td class="table-cell-custom text-white/90">{C['afr'].get(r, 0.0):.1f}</td>
                    <td class="table-cell-custom text-white/90">{C['tsp'].get(r, 0.0):.1f}</td>
                    <td class="table-cell-custom text-white/90">{osr:.1f}</td>
                    <td class="table-cell-custom text-white/90">{C['dsr'].get(r, 0.0):.1f}</td>
                    <td class="table-cell-custom text-white/90">{C['efr'].get(r, 0.0):.1f}</td>
                    <td class="table-cell-custom text-white/40">{svc:.1f}</td>
                    <td class="table-cell-custom text-white/90">{rs:.1f}</td>
                    <td class="table-cell-custom text-white/90">{C['ta'].get(r, 0.0):.1f}</td>
                    {shifts_html}
                </tr>'''
        return out

    def narrative_items(keys, alliance_color, data):
        bullets = [
            random.choice(AUTO_WIN_PHRASES if ((alliance_color=="red" and R["auto"]>=B["auto"]) or (alliance_color=="blue" and B["auto"]>R["auto"])) else AUTO_LOSE_PHRASES).format(name=alliance_color.title()),
            random.choice(ACTIVE_SHIFT_PHRASES).format(name=alliance_color.title(), active_shifts=data["active_shifts"], inactive_shifts=[s for s in range(1,5) if s not in data["active_shifts"]]),
            random.choice(ACTIVE_STRENGTH_PHRASES).format(predicted_active_fuel=data["predicted_active_fuel"]),
            random.choice(TSP_EFR_PHRASES).format(total=sum(C["tsp"].get(k,0) for k in keys)+sum(C["efr"].get(k,0) for k in keys), tsp=sum(C["tsp"].get(k,0) for k in keys), efr=sum(C["efr"].get(k,0) for k in keys))
        ]
        return "".join(f'<p class="mb-3 before:content-[\'>\'] before:mr-2 before:opacity-50"> {b}</p>' for b in bullets)

    def get_player_roles_html(keys, alliance_name, text_color, glow_class, border_color):
        cards_html = ""
        for r in keys:
            stats = {
                'Auto Specialist': C['afr'].get(r, 0.0),
                'Offensive Carry': C['osr'].get(r, 0.0),
                'Defensive Anchor': C['dsr'].get(r, 0.0),
                'Endgame Closer': C['efr'].get(r, 0.0),
                'Tower Focus': C['ta'].get(r, 0.0)
            }
            best_role = max(stats, key=stats.get)
            if stats[best_role] < 0.5: best_role = "Flex Support"
            overall_impact = sum(stats.values())
            cards_html += f"""
            <div class="bg-[#1c1b1e]/40 p-4 border-l-2 {border_color} flex justify-between items-center mb-2 hover:bg-[#1c1b1e] transition-colors">
                <div>
                    <div class="font-headline font-bold text-sm {text_color} {glow_class} tracking-widest">{r}</div>
                    <div class="font-mono text-[10px] text-white/50 uppercase tracking-widest mt-1">{best_role}</div>
                </div>
                <div class="text-right">
                    <div class="font-mono text-lg text-white/90">{overall_impact:.1f}</div>
                    <div class="font-mono text-[8px] text-white/30 uppercase">Impact Rating</div>
                </div>
            </div>
            """
        return f'<div class="flex-1"><h4 class="font-headline font-bold text-xs {text_color} uppercase tracking-widest mb-4">{alliance_name} Roster</h4>{cards_html}</div>'
    
    html_body = f"""
    <main class="max-w-6xl mx-auto pt-10 pb-20 px-6 relative z-10">
        
        <div onclick="window.location.href='scout://team/{team_selected}'" class="cursor-pointer text-secondary mb-6 font-mono text-xs hover:text-white transition-colors flex items-center gap-2 w-fit">
            <span>&larr;</span> BACK TO SCHEDULE
        </div>

        <header class="flex justify-between items-center border-b border-white/10 pb-4 mb-10">
            <div class="flex items-center gap-4">
                <div class="w-8 h-8 bg-white/10 flex items-center justify-center rounded-sm">
                    <span class="font-mono text-tertiary text-lg">*</span>
                </div>
                <h1 class="font-headline font-bold text-2xl text-white tracking-widest">{match_selected.split('_')[-1].upper()}</h1>
            </div>
            <div class="text-right">
                <span class="font-mono text-[10px] text-tertiary tracking-[0.2em] uppercase block">Scout Analytics</span>
                <span class="font-headline text-xs text-white/40 tracking-widest uppercase">{event_label}</span>
            </div>
        </header>

        <section class="relative mb-12">
            <div class="absolute -top-20 -right-20 w-96 h-96 bg-secondary-container/10 blur-[120px] rounded-full pointer-events-none"></div>
            <div class="flex items-center gap-2 mb-3">
                <span class="w-2 h-2 bg-white/80 animate-pulse rounded-full"></span>
                <span class="font-label text-xs tracking-[0.1em] text-white/60 uppercase font-bold">Live Projection</span>
            </div>
            <h2 class="font-headline text-5xl md:text-7xl font-bold tracking-tight uppercase leading-none {w_color_class} mb-4">
                {winner} WINS {cl(margin)}
            </h2>
            <p class="font-mono text-white/50 text-sm">{outcome}</p>

            <div class="mt-12 flex flex-col md:flex-row justify-between items-end border-l-2 border-white/10 pl-6 py-2">
                <div>
                    <span class="font-headline text-xs text-white/40 uppercase tracking-[0.2em] mb-2 block">Projected Scoreboard</span>
                    <div class="flex items-baseline gap-6">
                        <span class="font-headline text-6xl font-black {'text-secondary glow-red' if R['score'] >= B['score'] else 'text-white/40'} leading-none">{R['score']:.0f}</span>
                        <span class="font-headline text-2xl font-bold text-white/20 italic">VS</span>
                        <span class="font-headline text-6xl font-black {'text-tertiary glow-cyan' if B['score'] > R['score'] else 'text-white/40'} leading-none">{B['score']:.0f}</span>
                    </div>
                </div>
                <div class="mt-6 md:mt-0 text-right">
                    <span class="font-mono text-[10px] text-white/40 uppercase tracking-widest block mb-1">Strategic Edge</span>
                    <span class="font-body text-sm text-tertiary/80">{strategic}</span>
                </div>
            </div>
        </section>

        <div class="flex justify-between items-end mb-4 border-b border-white/10 pb-2">
            <h3 class="font-headline text-xs uppercase tracking-[0.2em] text-white/50 font-bold">Advanced Metrics</h3>
            <span class="font-mono text-[10px] text-tertiary/60">REF: SYS.V3</span>
        </div>
        
        <section class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-12">
            <div class="bg-[#0e0e10] p-6 border border-white/5 relative overflow-hidden">
                <p class="font-headline text-[10px] text-white/40 uppercase tracking-widest mb-6">Tactical System Strength (TSS)</p>
                <div class="flex justify-between items-end">
                    <div class="text-center">
                        <div class="font-mono text-3xl font-bold text-secondary glow-red mb-1">{R['tss']:.1f}</div>
                        <div class="font-mono text-[9px] text-white/30 uppercase">Red</div>
                    </div>
                    <div class="w-[1px] h-10 bg-white/10"></div>
                    <div class="text-center">
                        <div class="font-mono text-3xl font-bold text-tertiary glow-cyan mb-1">{B['tss']:.1f}</div>
                        <div class="font-mono text-[9px] text-white/30 uppercase">Blue</div>
                    </div>
                </div>
            </div>

            <div class="bg-[#0e0e10] p-6 border border-white/5 relative overflow-hidden">
                <p class="font-headline text-[10px] text-white/40 uppercase tracking-widest mb-6">Efficiency (PHI)</p>
                <div class="flex justify-between items-end">
                    <div class="text-center">
                        <div class="font-mono text-3xl font-bold text-secondary glow-red mb-1">{R['phi']:.2f}</div>
                        <div class="font-mono text-[9px] text-white/30 uppercase">Red</div>
                    </div>
                    <div class="w-[1px] h-10 bg-white/10"></div>
                    <div class="text-center">
                        <div class="font-mono text-3xl font-bold text-tertiary glow-cyan mb-1">{B['phi']:.2f}</div>
                        <div class="font-mono text-[9px] text-white/30 uppercase">Blue</div>
                    </div>
                </div>
            </div>

            <div class="bg-[#0e0e10] p-6 border border-white/5 relative overflow-hidden">
                <p class="font-headline text-[10px] text-white/40 uppercase tracking-widest mb-6">High-Impact Probability (HIP)</p>
                <div class="flex justify-between items-end">
                    <div class="text-center">
                        <div class="font-mono text-3xl font-bold text-secondary glow-red mb-1">{(R['hip']*100):.1f}<span class="text-sm opacity-50">%</span></div>
                        <div class="font-mono text-[9px] text-white/30 uppercase">Red</div>
                    </div>
                    <div class="w-[1px] h-10 bg-white/10"></div>
                    <div class="text-center">
                        <div class="font-mono text-3xl font-bold text-tertiary glow-cyan mb-1">{(B['hip']*100):.1f}<span class="text-sm opacity-50">%</span></div>
                        <div class="font-mono text-[9px] text-white/30 uppercase">Blue</div>
                    </div>
                </div>
            </div>
        </section>

        <section class="bg-gradient-to-b from-[#1c1b1e]/40 to-transparent p-6 border border-white/5 mb-12">
            <h3 class="font-headline text-xs uppercase tracking-[0.2em] text-white/50 font-bold text-center mb-8">Alliance Pulse / Shift Activation</h3>
            <div class="flex flex-col md:flex-row gap-12">
                <div class="flex-1">
                    <div class="text-center mb-6">
                        <span class="font-headline font-bold text-secondary text-sm glow-red tracking-widest uppercase">Red Alliance</span>
                        <div class="font-mono text-[9px] text-white/30 mt-2">ACTIVE: {R['active_shifts']}</div>
                    </div>
                    {shift_bars(R, 'red')}
                </div>
                <div class="w-[1px] bg-white/5 hidden md:block"></div>
                <div class="flex-1">
                    <div class="text-center mb-6">
                        <span class="font-headline font-bold text-tertiary text-sm glow-cyan tracking-widest uppercase">Blue Alliance</span>
                        <div class="font-mono text-[9px] text-white/30 mt-2">ACTIVE: {B['active_shifts']}</div>
                    </div>
                    {shift_bars(B, 'blue')}
                </div>
            </div>
        </section>

        <h3 class="font-headline text-xs uppercase tracking-[0.2em] text-white/50 font-bold mb-4">Combatant Systems</h3>
        <section class="bg-[#0e0e10] p-1 border border-white/5 overflow-x-auto mb-12">
            <table class="w-full whitespace-nowrap">
                <thead>
                    <tr>
                        <th class="table-header-custom text-left pl-2">Robot</th>
                        <th class="table-header-custom">AFR</th>
                        <th class="table-header-custom">TSP</th>
                        <th class="table-header-custom">OSR</th>
                        <th class="table-header-custom">DSR</th>
                        <th class="table-header-custom">EFR</th>
                        <th class="table-header-custom">SVC</th>
                        <th class="table-header-custom">RS</th>
                        <th class="table-header-custom">TA</th>
                        <th class="table-header-custom">S1</th>
                        <th class="table-header-custom">S2</th>
                        <th class="table-header-custom">S3</th>
                        <th class="table-header-custom">S4</th>
                    </tr>
                </thead>
                <tbody>
                    {robot_rows(red_keys, '#ffb3ae')}
                    <tr><td colspan="13" class="h-2"></td></tr>
                    {robot_rows(blue_keys, '#74f5ff')}
                </tbody>
            </table>
        </section>

        <section class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            <div class="bg-[#1c1b1e]/60 p-6 border-l-2 border-secondary/50 backdrop-blur-md">
                <h4 class="font-headline font-bold text-xs text-secondary uppercase tracking-widest mb-4">Red Narrative</h4>
                <div class="font-mono text-xs leading-relaxed text-white/70">
                    {narrative_items(red_keys, 'red', R)}
                </div>
            </div>
            <div class="bg-[#1c1b1e]/60 p-6 border-l-2 border-tertiary/50 backdrop-blur-md">
                <h4 class="font-headline font-bold text-xs text-tertiary uppercase tracking-widest mb-4">Blue Narrative</h4>
                <div class="font-mono text-xs leading-relaxed text-white/70">
                    {narrative_items(blue_keys, 'blue', B)}
                </div>
            </div>
        </section>

        <h3 class="font-headline text-xs uppercase tracking-[0.2em] text-white/50 font-bold mb-4">Key Players & Assigned Roles</h3>
        <section class="flex flex-col md:flex-row gap-6 mb-12 bg-[#0e0e10] p-6 border border-white/5">
            {get_player_roles_html(red_keys, 'Red Alliance', 'text-secondary', 'glow-red', 'border-secondary/50')}
            <div class="w-[1px] bg-white/5 hidden md:block"></div>
            {get_player_roles_html(blue_keys, 'Blue Alliance', 'text-tertiary', 'glow-cyan', 'border-tertiary/50')}
        </section>

    </main>
    </body>
    </html>
    """

    return get_html_head() + html_body


# ─── PyQt Thread Workers ──────────────────────────────────────────────────────
class SetupWorker(QThread):
    finished = pyqtSignal(dict)
    def run(self):
        wm = load_event_weeks()
        self.finished.emit(wm)

class LoaderWorker(QThread):
    finished = pyqtSignal(list)
    def __init__(self, func, *args):
        super().__init__()
        self.func = func
        self.args = args
    def run(self):
        res = self.func(*self.args)
        self.finished.emit(res)

class LeaderboardWorker(QThread):
    resultReady = pyqtSignal(str)
    def __init__(self, ev_label, ev_key):
        super().__init__()
        self.ev_label = ev_label
        self.ev_key = ev_key
    def run(self):
        html = build_leaderboard_html(self.ev_label, self.ev_key)
        self.resultReady.emit(html)

class ScheduleWorker(QThread):
    resultReady = pyqtSignal(str)
    def __init__(self, ev_label, ev_key, team_key):
        super().__init__()
        self.ev_label = ev_label
        self.ev_key = ev_key
        self.team_key = team_key
    def run(self):
        html = build_schedule_html(self.ev_label, self.ev_key, self.team_key)
        self.resultReady.emit(html)

class AnalysisWorker(QThread):
    resultReady = pyqtSignal(str)
    def __init__(self, ev_label, ev_key, t_key, m_key):
        super().__init__()
        self.ev_label = ev_label
        self.ev_key = ev_key
        self.t_key = t_key
        self.m_key = m_key
        
    def run(self):
        # Pass both ev_label and ev_key here!
        html = build_html_report(self.ev_label, self.ev_key, self.t_key, self.m_key)
        self.resultReady.emit(html)

# ─── Custom WebEnginePage for interception ────────────────────────────────────
class WebAppPage(QWebEnginePage):
    navigation_requested = pyqtSignal(str)
    
    def acceptNavigationRequest(self, url, _type, isMainFrame):
        if url.scheme() == "scout":
            self.navigation_requested.emit(url.toString())
            return False
        return super().acceptNavigationRequest(url, _type, isMainFrame)

# ─── PyQt GUI ─────────────────────────────────────────────────────────────────
APP_STYLE = """
    QMainWindow { background-color: #08080A; }
    QWidget { font-family: "Segoe UI", sans-serif; color: #e5e1e4; }
    QFrame#Sidebar { background-color: #0e0e10; border-right: 1px solid #1c1b1e; }
    QLabel#LogoStar { color: #74f5ff; font-size: 24px; }
    QLabel#LogoText { font-size: 18px; font-weight: bold; }
    QLabel#LogoHighlight { color: #74f5ff; font-size: 18px; font-weight: bold; }
    QLabel#SectionLabel { color: #5d5f5f; font-family: "Consolas", monospace; font-size: 10px; font-weight: bold; text-transform: uppercase; letter-spacing: 2px; margin-top: 15px; }
    QComboBox { background-color: #1c1b1e; border: 1px solid #353437; border-radius: 0px; padding: 8px; color: #e5e1e4; }
    QComboBox:disabled { background-color: #08080A; border: 1px solid #1c1b1e; color: #353437; }
    QComboBox::drop-down { border: none; width: 20px; }
    QComboBox QAbstractItemView { background-color: #1c1b1e; border: 1px solid #353437; selection-background-color: #353437; color: #e5e1e4; }
    QPushButton#LoadBtn { background-color: #74f5ff; color: #000000; border: none; border-radius: 0px; padding: 14px; font-weight: bold; font-size: 13px; text-transform: uppercase; letter-spacing: 1px; }
    QPushButton#LoadBtn:disabled { background-color: #1c1b1e; color: #5d5f5f; }
    QPushButton#LoadBtn:hover:!disabled { background-color: #a5f8ff; }
    QLabel#StatusLabel { color: #5d5f5f; font-size: 11px; margin-top: 10px; }
"""

class ScoutApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Scout Analytics")
        self.resize(1400, 900)
        
        self.event_map = {}
        self.current_team = None
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(280)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(25, 30, 25, 30)
        
        logo_layout = QHBoxLayout()
        star = QLabel("*")
        star.setObjectName("LogoStar")
        text1 = QLabel("Scout ")
        text1.setObjectName("LogoText")
        text2 = QLabel("Analytics")
        text2.setObjectName("LogoHighlight")
        logo_layout.addWidget(star)
        logo_layout.addWidget(text1)
        logo_layout.addWidget(text2)
        logo_layout.addStretch()
        sidebar_layout.addLayout(logo_layout)
        
        def add_combo(label_text):
            lbl = QLabel(label_text)
            lbl.setObjectName("SectionLabel")
            cb = QComboBox()
            cb.setEnabled(False)
            sidebar_layout.addWidget(lbl)
            sidebar_layout.addWidget(cb)
            return cb

        self.cb_week = add_combo("WEEK")
        self.cb_event = add_combo("EVENT")
        
        sidebar_layout.addSpacing(30)
        
        self.btn_load = QPushButton("Load Event")
        self.btn_load.setObjectName("LoadBtn")
        self.btn_load.setEnabled(False)
        sidebar_layout.addWidget(self.btn_load)
        
        self.lbl_status = QLabel("System Initializing...")
        self.lbl_status.setObjectName("StatusLabel")
        self.lbl_status.setWordWrap(True)
        sidebar_layout.addWidget(self.lbl_status)
        
        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)
        
        self.web_view = QWebEngineView()
        self.web_page = WebAppPage(self.web_view)
        self.web_page.navigation_requested.connect(self.handle_navigation)
        self.web_view.setPage(self.web_page)
        
        placeholder_html = """
        <html>
        <body style='background:#08080A; color:#e5e1e4; font-family:"Segoe UI", sans-serif; display:flex; justify-content:center; align-items:center; height:100vh; margin:0;'>
            <div style='text-align:center;'>
                <div style='color:#74f5ff; font-family:Consolas, monospace; font-size:12px; letter-spacing:2px; margin-bottom:10px;'>SYSTEM OFFLINE</div>
                <h1 style='color:#ffffff; font-size:24px; text-transform:uppercase; letter-spacing:1px;'>Awaiting Event Selection</h1>
            </div>
        </body>
        </html>
        """
        self.web_view.setHtml(placeholder_html)
        main_layout.addWidget(self.web_view, 1)
        
        self.cb_week.currentTextChanged.connect(self.on_week)
        self.cb_event.currentTextChanged.connect(lambda: self.btn_load.setEnabled(bool(self.cb_event.currentText())))
        self.btn_load.clicked.connect(self.load_leaderboard)

        self.setup_worker = SetupWorker()
        self.setup_worker.finished.connect(self.bootstrap_done)
        self.setup_worker.start()

    def bootstrap_done(self, wm):
        self.cb_week.setEnabled(True)
        weeks = sorted(wm.keys(), key=lambda v: int(v))
        if weeks:
            self.cb_week.addItems(weeks)
            self.lbl_status.setText("Parameters loaded.")
        else:
            self.lbl_status.setText("Data connection failed.")

    def set_loading(self, message):
        self.lbl_status.setText(message)
        self.lbl_status.setStyleSheet("color: #74f5ff; font-size: 11px; margin-top: 10px;")

    def clear_loading(self, message):
        self.lbl_status.setText(message)
        self.lbl_status.setStyleSheet("color: #5d5f5f; font-size: 11px; margin-top: 10px;")

    def show_web_loading(self):
        loading_html = """
        <html>
        <body style='background:#08080A; display:flex; justify-content:center; align-items:center; height:100vh; margin:0;'>
            <div style='text-align:center;'>
                <div style='color:#74f5ff; font-family:Consolas, monospace; font-size:12px; letter-spacing:2px; margin-bottom:15px; animation: pulse 1.5s infinite;'>PROCESSING DATA CUBE</div>
                <div style='width: 200px; height: 2px; background: #1c1b1e; margin: 0 auto; overflow: hidden;'>
                    <div style='width: 50%; height: 100%; background: #74f5ff; transform: translateX(-100%); animation: load 1s infinite linear;'></div>
                </div>
            </div>
            <style>
                @keyframes load { 0% { transform: translateX(-100%); } 100% { transform: translateX(200%); } }
                @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
            </style>
        </body>
        </html>
        """
        self.web_view.setHtml(loading_html)

    def on_week(self, wk):
        if not wk: return
        self.set_loading(f"Querying Week {wk} events...")
        self.cb_event.clear()
        self.cb_event.setEnabled(False)
        self.btn_load.setEnabled(False)
        
        self.w1 = LoaderWorker(load_events_for_week, wk)
        self.w1.finished.connect(self.events_loaded)
        self.w1.start()

    def events_loaded(self, evs):
        self.event_map.clear()
        if not evs:
            self.clear_loading("No events found.")
            return
        labels = []
        for ev in evs:
            loc = getattr(ev, 'location_name', '') or ''
            lbl = f"{getattr(ev, 'name', '')} ({loc})" if loc else getattr(ev, 'name', ev.key)
            labels.append(lbl)
            self.event_map[lbl] = ev.key
        self.cb_event.setEnabled(True)
        self.cb_event.addItems(labels)
        self.clear_loading(f"Secured {len(labels)} event records.")

    def handle_navigation(self, url_str):
        # Format: scout://action/id
        url = url_str.replace("scout://", "")
        parts = url.split('/')
        action = parts[0]
        
        if action == "leaderboard":
            self.load_leaderboard()
        elif action == "team" and len(parts) > 1:
            self.current_team = parts[1]
            self.load_schedule(self.current_team)
        elif action == "match" and len(parts) > 1:
            self.load_match(parts[1])

    def load_leaderboard(self):
        ev_label = self.cb_event.currentText()
        ev_key = self.event_map.get(ev_label)
        if not ev_key: return
        
        self.set_loading("Executing analytical models for event...")
        self.show_web_loading()
        self.w_lb = LeaderboardWorker(ev_label, ev_key)
        self.w_lb.resultReady.connect(self.view_loaded)
        self.w_lb.start()

    def load_schedule(self, team_key):
        ev_label = self.cb_event.currentText()
        ev_key = self.event_map.get(ev_label)
        if not ev_key: return
        
        self.set_loading(f"Loading schedule for {team_key}...")
        self.show_web_loading()
        self.w_sch = ScheduleWorker(ev_label, ev_key, team_key)
        self.w_sch.resultReady.connect(self.view_loaded)
        self.w_sch.start()

    def load_match(self, match_key):
        ev_label = self.cb_event.currentText()
        ev_key = self.event_map.get(ev_label)
        if not ev_key or not self.current_team: return
        
        self.set_loading(f"Analyzing match {match_key}...")
        self.show_web_loading()
        self.w_match = AnalysisWorker(ev_label, ev_key, self.current_team, match_key)
        self.w_match.resultReady.connect(self.view_loaded)
        self.w_match.start()

    def view_loaded(self, html):
        self.web_view.setHtml(html)
        self.clear_loading("System ready.")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    window = ScoutApp()
    window.show()
    sys.exit(app.exec())