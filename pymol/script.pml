set valence, 0
select IS1, organic
hide cartoon
zoom IS1
remove hydrogens
util.cbac('IS1')
zoom act_site
select act_site, byres IS1 around 3.5
set stick_color, white, act_site
show sticks, act_site
zoom act_site
label act_site and name CB, resn+resi
distance pol_cont, IS1, act_site, mode=2
hide label
bg_color white
set mesh_radius = 0.01
set antialias = 1
set stick_radius = 0.22
set dash_radius = 0.07
set sphere_scale = 0.22
set ribbon_radius = 0.1
set direct = 0.0
set cartoon_fancy_helices = 1
bg_color white
set gamma = 1.5
util.ray_shadows('none')
set ray_trace_mode = 1
set cartoon_flat_sheets = 1.0
set cartoon_smooth_loops = 0
cartoon automatic
show cartoon
hide nonbonded
hide lines
set opaque_background, off   # or on if you want background
ray

