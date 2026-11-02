Scenario definitions used in the configs (copied into each file so that every
config is self-contained):

intersection        benchmark settings: 4 CAVs (one per approach), 10 human-driven
                    vehicles, as in CoDrivingLLM (Fang et al., 2025)
intersection_dense  8 CAVs (two per approach), 14 human-driven vehicles, new
                    (no vehicles are added during the episode)
merge               on-ramp merge, 4 CAVs (2 main road, 2 ramp), 6 human-driven vehicles
                    following IDM only (they do not stop to yield at the merge),
                    50 s limit

See any of them on screen with  python experiments/watch.py --scenario <name>
