# Pearlie look mechanics

Pearlie is a petite humanoid pixel character. Her lower body, feet, dress hem, and the lower edge of the hair mass stay anchored to the same baseline while the gaze travels. The eyes and eyelids lead the motion; the nose, chin, neck, and upper torso follow with restrained near-rigid pixel redraws. The long hair follows the head turn only through small side-occlusion and strand shifts, preserving its volume and silhouette. The cardigan and pink dress are worn clothing, so they stay body-locked; Pearlie has no held prop that should lag or swing.

Use the original illustrated eye construction: move the eye surface, iris, pupil, eyelid, and highlight together as one drawn eye shape rather than sliding a new pupil over a fixed white. Do not add replacement or googly eyes. Preserve the same warm skin, dark-brown outline, long black hair, cream cardigan, pale-pink dress, heart chest detail, socks, and shoes in every pose.

## Cardinal pose families (viewer/screen coordinates)

- `000` up / 12 o'clock: both eyes and the nose aim above the head center; upper eyelids lift slightly, chin tips up, and the torso follows only a little. Feet, dress hem, and lower hair remain planted.
- `090` screen-right: nose tip and both eye centers move to the right of the head center; the face turns right so the right cheek/side becomes more visible and the opposite hair edge occludes slightly. The lower body remains front-registered.
- `180` down / 6 o'clock: eyes and nose aim below the head center; upper lids lower, chin dips, and the head/neck settle slightly forward. Hair may overlap the upper shoulders a little, without changing body scale.
- `270` screen-left: nose tip and both eye centers move to the left of the head center; the face turns left so the left cheek/side becomes more visible and the opposite hair edge occludes slightly. It must visibly oppose `090`; the lower body remains front-registered.

The diagonal poses interpolate these four families in fixed clockwise order. Row 9 travels `000 -> 090 -> 180` through its eight evenly spaced steps, and row 10 continues `180 -> 270 -> 000`. Each 22.5-degree step makes a comparable change in eye direction, eyelid shape, nose/head turn, and small hair follow-through. The boundaries `157.5 -> 180`, `337.5 -> 000`, and row 10 -> row 9 must be continuous.

## Motion budget and prohibitions

Keep the sprite's feet/base, lower torso, scale, and baseline stable. Use small eye/eyelid changes first, then a restrained head and upper-body follow-through; never rotate, skew, affine-tilt, or broadly warp the whole sprite to fake looking. Do not mirror individual direction cells, add new facial features, introduce text/labels/arrows, or add shadows, glows, scenery, or detached effects. Every direction must be distinguishable from Pearlie's neutral idle face at normal pet size while remaining one coherent pixel-art animation family.
