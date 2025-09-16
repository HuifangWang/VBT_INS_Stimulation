# Domain Description 1.1 	

Interfaces 3
Interface InnerSkull: "PATH/inner_skull.tri"
Interface OuterSkull: "PATH/outer_skull.tri"
Interface OuterSkin: "PATH/outer_skin.tri"

Domains 4
Domain Brain: -InnerSkull 
Domain Skull: -OuterSkull +InnerSkull
Domain Skin: -OuterSkin +OuterSkull
Domain Air: +OuterSkin