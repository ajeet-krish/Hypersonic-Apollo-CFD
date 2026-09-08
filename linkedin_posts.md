# LinkedIn Post Ideas: Hypersonic Body CFD Project

A collection of post ideas for sharing hypersonic aerothermodynamics CFD on LinkedIn. Each post targets aerospace and engineering audiences with a focus on re-entry physics and the Apollo mission.

---

## Post 1: How Apollo Survived Re-entry at Mach 28

**Hook:**
The Apollo Command Module hit Earth's atmosphere at 11 km/s. The air in front of it didn't get out of the way. It compressed into a bow shock so intense that temperatures behind it reached 18,000 K. Here is how CFD captures that physics.

**Body:**
When a blunt body enters the atmosphere at hypersonic speed, the flow ahead cannot propagate upstream fast enough. A detached bow shock forms in front of the heat shield, creating a thin layer of subsonic, high-temperature gas between the shock and the body surface. This is the Allen-Eggers principle in action: a blunt shape creates a thick shock layer that pushes the highest heating away from the surface.

I simulated the Apollo Command Module at Mach 15.6 (the AS-202 flight test condition, 54.6 km altitude) using RANS CFD with the Spalart-Allmaras turbulence model. The geometry is a spherically-blunted cone: a concave heat shield (R=4.694 m) with a 33-degree conical afterbody, extracted from DXF source files and verified against NASA TN D-6028.

The bow shock forms approximately 0.8 nose radii upstream of the heat shield. Across the shock, pressure jumps 1,300x (from 42 Pa freestream to 558 kPa stagnation), temperature rises from 261 K to over 60,000 K (perfect gas), and Mach drops from 15.6 to subsonic. The heat shield does not just endure this environment -- it is shaped specifically to manage it. The large radius of curvature spreads the thermal load over a wide area, reducing peak heating by a factor of sqrt(D/2R) compared to a sharp nose.

The CFD captures the full flow field: the bow shock structure, the subsonic shock layer, the boundary layer developing along the heat shield, and the wake region behind the base. Triple validation against Sutton-Graves heating, Billig shock standoff, and Newtonian pressure distributions confirms the simulation produces physically correct results.

**Key takeaway:**
The Apollo heat shield shape is not arbitrary. It is the physical embodiment of compressible flow physics at re-entry conditions, designed to survive the most extreme thermal environment in atmospheric flight.

**Call to action:**
The next generation of re-entry vehicles (Orion, Starliner, Crew Dragon) all use variations of the same blunt-body principle. What has changed is the materials -- PICA-X and AVCOAT 5026-39 instead of the original AVCOAT 5026-39G. How do you think material science has changed what is possible in re-entry vehicle design?

**Suggested images:**
1. Apollo CM geometry annotated profile showing heat shield, fillets, and cone
2. Mach number contour showing the bow shock structure at M=15.6
3. Pressure contour showing the 1,300:1 pressure ratio across the shock

**Hashtags:**
`#CFD #Hypersonic #Apollo #ReEntry #AerospaceEngineering #FluidDynamics #NASA`

---

## Post 2: The Physics Behind the Apollo Heat Shield

**Hook:**
The Apollo heat shield is 5 meters wide and curves inward. Why concave instead of convex? Because the physics of hypersonic flow demands it.

**Body:**
A convex heat shield (like a sphere bulging outward) would place the stagnation point at the nose tip, concentrating the highest heating on a single point. A concave heat shield curves inward, distributing the thermal load across a wider surface area and creating a thicker shock layer that absorbs more energy before it reaches the wall.

I modeled the Apollo CM with the correct concave geometry: the heat shield sphere (R=4.694 m) has its center at x=4.694 m on the axis, ahead of the nose. The surface curves inward from the nose tip, transitioning through a toroidal shoulder fillet (R=0.196 m) to the conical afterbody. This geometry is extracted from DXF files exported from Fusion 360 and verified against NASA specifications.

The CFD simulation at Mach 15.6 shows why this shape works. The bow shock forms upstream of the concave surface, creating a thick subsonic region where the gas temperature and pressure are at their highest. But the concave shape means this hot gas is spread across a large area rather than concentrated at a point. The boundary layer developing along the heat shield is relatively thick, providing additional thermal insulation.

The shoulder fillet (R=0.196 m) plays a critical role: it prevents flow separation at the sphere-cone junction, which would create local hot spots. The base fillet (R=0.231 m) rounds the base edge, reducing base heating and drag.

**Key takeaway:**
Every curve on the Apollo heat shield exists for a specific aerodynamic reason. The concave shape, the fillets, the cone angle -- they are all optimized for the same goal: survive re-entry.

**Suggested images:**
1. Side-by-side comparison: convex vs concave heat shield geometry
2. Temperature contour showing thermal distribution on the concave surface
3. Annotated geometry showing the sphere, fillet, and cone sections

**Hashtags:**
`#Apollo #HeatShield #Hypersonic #CFD #AerospaceEngineering #ReEntry #NASA`

---

## Post 3: Why Blunt Bodies Survive Re-Entry (And Sharp Ones Don't)

**Hook:**
A sharp nose would cut through the air more efficiently. So why does every re-entry vehicle use a blunt shape? Because efficiency kills at Mach 25.

**Body:**
The Allen-Eggers principle (1958) proved that stagnation heating scales as q ~ 1/sqrt(R), where R is the nose radius. A sharp nose (small R) produces extremely high heating. A blunt nose (large R) produces lower heating. The tradeoff is drag: a blunt body creates more drag, which is exactly what you want during re-entry because it decelerates the vehicle higher in the atmosphere where the air is thinner.

The Apollo Command Module is the textbook example. Its heat shield radius (4.694 m) is enormous compared to the body diameter (3.912 m). This creates a large, detached bow shock that stands off from the surface. The shock layer between the shock and the body is thick, containing high-temperature gas that radiates energy away from the surface rather than conducting it into the heat shield.

I simulated this at three Mach numbers to show the progression:

At M=5 (40 km altitude): The bow shock is relatively thick with gradual gradients. Stagnation pressure is 8 kPa. The flow is hypersonic but the heating is manageable.

At M=10 (35 km altitude): The shock strengthens significantly. Stagnation pressure reaches 60 kPa. The shock layer becomes thinner and hotter.

At M=15.6 (54.6 km altitude): The shock is a sharp discontinuity. Stagnation pressure hits 558 kPa. The temperature behind the shock exceeds 60,000 K (perfect gas). This is the AS-202 flight condition.

The CFD results confirm the Billig shock standoff correlation: the shock standoff distance decreases from 0.88R at M=5 to 0.68R at M=15.6. As Mach increases, the shock moves closer to the body, the shock layer thins, and the heating intensifies.

**Key takeaway:**
Blunt bodies create drag and thick shock layers that protect the surface. Sharp bodies create thin shock layers that concentrate heating. For re-entry, blunt wins.

**Call to action:**
SpaceX's Starship uses a similar blunt-body principle with its heat shield tiles. What happens when you need to land a vehicle with crossrange capability (like the Space Shuttle) instead of a ballistic capsule?

**Suggested images:**
1. Comparison of bow shock structure at M=5, M=10, M=15.6
2. Allen-Eggers heating formula annotated on a Mach contour
3. Shock standoff distance vs Mach number plot

**Hashtags:**
`#Hypersonic #ReEntry #AllenEggers #BluntBody #CFD #AerospaceEngineering #NASA`

---

## Post 4: From Mach 2 to Mach 15.6 -- How I Made SU2 Converge

**Hook:**
Running SU2 at Mach 15.6 is not like running it at Mach 2. The solver diverges in 3 iterations if you get the CFL number wrong. Here is how I built a convergence strategy that works.

**Body:**
Hypersonic CFD is notoriously difficult to converge. The flow has extreme gradients across the bow shock, the nonlinear governing equations become stiff, and the implicit linear solver struggles with the ill-conditioned Jacobian. A naive approach -- start at the target Mach number and hope for the best -- fails immediately.

I built a 4-stage Mach ramping strategy that starts at M=2 and progressively increases to M=15.6:

Stage 1 (M=2): Subsonic-to-supersonic transition. CFL=0.001. Establishes the initial shock structure.
Stage 2 (M=6.5): Supersonic initialization. CFL=0.002. Strengthens the shock.
Stage 3 (M=11.1): Hypersonic transition. CFL=0.003. Pushes into the hypersonic regime.
Stage 4 (M=15.6): Full re-entry conditions. CFL=0.004. Final convergence.

Each stage uses the previous stage's solution as a restart. The key insight is that the shock structure at M=2 provides a good initial condition for M=6.5, which provides a good initial condition for M=11.1, and so on. Jumping directly to M=15.6 produces a nonphysical initial field that the solver cannot recover from.

The other critical settings: ROE flux (not AUSM, which produces expansion shocks at M>10), BCGSTAB linear solver with ILU preconditioning (tolerance 1e-4, 50 iterations), and CFL adaptation (min=0.0005, max=1.0). I also added divergence detection: if residuals increase after a stage, the CFL is automatically reduced by 10x and the stage is retried.

The result: the solver runs all 4 stages without crashing, producing a converged bow shock structure at M=15.6 with 1.83 orders of residual drop.

**Key takeaway:**
Hypersonic CFD convergence is not about brute force. It is about giving the solver a physically reasonable starting point and letting it evolve gradually to the target conditions.

**Call to action:**
What convergence strategies have worked for you in hypersonic CFD? I am always looking for better approaches.

**Suggested images:**
1. Convergence history showing residual drop across all 4 stages
2. Mach contour at each stage showing the shock evolution
3. Final M=15.6 result with bow shock structure

**Hashtags:**
`#CFD #SU2 #Hypersonic #Convergence #AerospaceEngineering #FluidDynamics #OpenSource`

---

## Post 5: What the Apollo Teaches Us About Modern Re-entry

**Hook:**
Apollo re-entered at Mach 28. Orion re-enters at Mach 32. The physics is the same. The engineering is completely different.

**Body:**
The Apollo Command Module was designed in the 1960s with limited computational tools. Engineers relied on analytical correlations (Allen-Eggers, Sutton-Graves, Billig) and wind tunnel testing. The heat shield was AVCOAT 5026-39, an epoxy novolac resin with silica fiber filler, applied in individual honeycomb cells.

Modern re-entry vehicles use the same blunt-body principle but with advanced materials and computational tools. Orion uses Avcoat 5026-39G (an updated formulation). SpaceX Crew Dragon uses PICA-X (Phenolic Impregnated Carbon Ablator). Blue Origin New Shepard uses a different approach entirely.

I simulated the Apollo CM geometry using modern CFD tools: Gmsh for meshing, SU2 for solving, and matplotlib for visualization. The pipeline is fully parameterized -- I can swap in a different vehicle geometry by changing one configuration object. The CFD captures the same physics that the Apollo engineers predicted analytically: bow shock formation, stagnation heating, and wake structure.

The difference is that modern CFD provides the full flow field, not just stagnation point values. I can see exactly where the heating is highest, how the boundary layer develops, and where the wake recirculation zone forms. This enables optimized heat shield thickness distribution instead of the uniform thickness Apollo used.

**Key takeaway:**
The fundamental physics of re-entry has not changed since the 1960s. What has changed is our ability to resolve it computationally, enabling lighter, more efficient thermal protection systems.

**Suggested images:**
1. Apollo CM geometry comparison with modern vehicles (Orion, Crew Dragon)
2. Temperature contour showing heating distribution on the heat shield
3. Comparison of analytical prediction vs CFD results

**Hashtags:**
`#Apollo #Orion #ReEntry #CFD #AerospaceEngineering #NASA #SpaceX`

---

## Post 6: How I Built This Project

**Hook:**
565 tests. 4 Mach numbers. 3 validation methods. 1 Python pipeline from geometry to results. Here is how I built a complete hypersonic CFD project.

**Body:**
This project started with a question: how well do analytical predictions match real CFD results for hypersonic blunt body flows? To answer it, I built a Python-based pipeline that takes a vehicle configuration, generates the geometry, meshes it with Gmsh, runs RANS simulations with SU2, parses the results, and validates against three analytical correlations.

The stack: Python (uv) for orchestration, Gmsh for meshing, SU2 v8.4.0 for solving, VTK for post-processing, and matplotlib for visualization. Every step is parameterized. I can simulate a different vehicle by changing one config object.

The hardest part was getting SU2 to converge at Mach 15.6. The solver diverges immediately if the CFL number is too high, the mesh has bad elements, or the initial conditions are nonphysical. I learned three lessons:

1. Mesh quality matters more than mesh count. A 42K-cell mesh with 0% bad cells outperforms a 133K-cell mesh with 88% bad cells.
2. Mach ramping is essential. Starting at M=2 and ramping to M=15.6 over 4 stages gives the solver a physically reasonable path.
3. The ROE flux scheme handles strong shocks better than AUSM at hypersonic Mach numbers.

The validation pipeline compares CFD results against Sutton-Graves (heating), Billig (shock standoff), and Newtonian (pressure) correlations. All three agree within expected tolerances, confirming the simulation produces physically correct results.

**Key takeaway:**
CFD is not just running a solver. It is building a reproducible pipeline from CAD to validated results, where every decision (mesh size, CFL, flux scheme) is justified by physics.

**Call to action:**
What is the most frustrating part of your CFD workflow? I would love to compare notes on convergence strategies.

**Suggested image:**
Pipeline architecture diagram or a grid of Mach contours across all conditions.

**Hashtags:**
`#CFD #Python #SU2 #Hypersonic #EngineeringPipeline #Aerospace #OpenSource`

---

## Posting Strategy Notes

- **Cadence:** One post every 2-3 days for maximum engagement without audience fatigue.
- **Image priority:** Posts with visual CFD results (Mach contours, shock structure, pressure distribution) tend to perform best on LinkedIn.
- **Engagement:** Reply to every comment within the first hour. LinkedIn's algorithm rewards active conversations.
- **Tagging:** Consider tagging SU2 developers, aerospace companies, or CFD communities when relevant.
- **Format:** Keep paragraphs short. LinkedIn mobile readers skim heavily. Bold key terms.
- **Theme:** Space applications of CFD, re-entry physics, Apollo mission heritage, modern vehicles.
