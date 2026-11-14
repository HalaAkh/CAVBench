Scenario definitions used in the configs (copied into each file so that every
config is self-contained):

intersection        benchmark settings: 4 CAVs (one per approach), as in CoDrivingLLM
                    (Fang et al., 2025); initial_vehicle_count 10 = 10 spawn attempts
                    (background vehicles with 60 % probability each), then every
                    background vehicle within 20 m of a CAV is removed: about 1-4
                    human-driven vehicles remain (2.65 on average on the tuning seeds)
intersection_dense  8 CAVs (two per approach), initial_vehicle_count 14, same rule:
                    about 0-4 human-driven vehicles remain (2.1 on average), new
                    (in both, no vehicles are added during the episode)
intersection_case2  only configs/mcts_case2_check.yaml: 4 CAVs + exactly 4 human-
                    driven vehicles, one per approach, 15-20 m ahead of the CAV
                    (hdv_per_approach: 1), as in Case 2 of Lin et al. (2025)
merge               on-ramp merge, 4 CAVs (2 main road, 2 ramp), 5-6 human-driven vehicles
                    (6 are tried; one is left out when 50 random tries find no spot
                    at least 15 m from the other vehicles; 5.7 on average on the tuning seeds)
                    following IDM only (they do not stop to yield at the merge),
                    50 s limit

See any of them on screen with  python experiments/watch.py --scenario <name>
