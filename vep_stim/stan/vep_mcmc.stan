data {
  int nn;
  int nt;
  int ns;
  
  
  matrix[nt, ns] Obs;
  matrix [nn, nn] SC;
  matrix[ns, nn] gain;
  matrix[nn,nn] eig;
}

transformed data {
  real dt;
  real tau0;
  real I1;
  real Ks;
  
  vector[nt*ns] xs;
  matrix[nt,ns] wtm;
  vector[nt*ns] wt;
  vector[nn] x_init;
  for(i in 1:nn){
    x_init[i]=-1.98;
  }
  
  xs=to_vector(log(Obs));
  for(i in 1:nt){
    wtm[i,]=Obs[i,:]-Obs[1,:];
  }
  wt=to_vector(wtm);
  
  
  dt=0.1;
  tau0=15.0; // because i use 500 time points (10.0 for 180 time points)
  I1=3.1;
  Ks=1.0;
  
  
  
  
  
}

parameters {
  
  vector[nn] z_init_star;
  vector[nn] x0_star;
  real <lower=-1.0> K_star;
  real <lower=0.0> amp_star;
  vector[ns] u_star;
  real log_eps_sq;
}

transformed parameters {
  
  vector[nn] x0;
  vector[nn] z_init;
  real K;
  real amp;
  vector[ns] u;
  real eps; 
  
  x0 = -3.0 + 0.30*eig*x0_star;
  z_init=4.1+0.1*eig*z_init_star;
  K =  Ks + 1.0*K_star;
  amp = 0.005 + 0.01*amp_star;
  u= 2.0+1.0*u_star;
  eps=exp(0.33*log_eps_sq-3.0); 
  //eps=exp(0.33*log_eps_sq-8.0); 
  
  
}

model {
  matrix[nn, nt] x;
  matrix[nn, nt] z;
  matrix[nt, ns] xhat2;
  vector[nt*ns] xhat;
  
  real dx;
  real dz;
  real gx;
  
  // "group-level (hierarchical) SD across betas"
  
  x0_star ~ normal(0.,1.);  
  amp_star ~ normal(0. ,1.);
  K_star ~ normal(0., 1.);  
  log_eps_sq ~ normal(0.,10.0);
  u_star ~ normal(0., 1.0);
  z_init_star ~ normal(0, 1.);
  
  /* integrate & predict */
    
    for (i in 1:nn) {
      x[i, 1] =  x_init[i];
      z[i, 1] =  z_init[i];
    } 
  
  for (t in 1:(nt-1)) {
    for (i in 1:nn) {
      gx = 0;
      for (j in 1:nn)
        gx = gx + SC[i, j]*(x[j, t] - x[i, t]);
      dx = 1.0 - x[i, t]*x[i, t]*x[i, t] - 2.0*x[i, t]*x[i, t] - z[i, t] + I1;
      dz = (1/tau0)*(4*(x[i, t] - x0[i]) - z[i, t] - K*gx);
      x[i, t+1] = x[i, t] + dt*dx ; 
      z[i, t+1] = z[i, t] + dt*dz ; 
    }
  }   
  
  for(i in 1:nt){
    xhat2[i,]=to_row_vector(amp * (gain * x[,i]) +  u);
  }
  
  xhat=to_vector(xhat2);
  
  for(i in (10*ns+1):(nt*ns)){
    target+=  normal_lpdf(xs[i]| xhat[i], eps);  
  }
}    

generated quantities {
  
  matrix[nn, nt] x;
  matrix[nn, nt] z;
  matrix[nt, ns] xhat_qqc2;
  matrix[nt,ns] xhat_q2;
  
  vector[nt*ns] xhat_qqc;
  vector[nt*ns] x_ppc;
  vector[nt*ns] log_lik;
  
  matrix[nt,ns] xhat_q;
  matrix[nt,ns] x_p;
  vector[nt] log_lik_sum = rep_vector(0,nt);
  
  real gx;
  real dx;
  real dz;
  
  int num_params;
  int num_data;
  num_params=2*(nt*ns)+nn+6;
  num_data=nt*ns;
  
  
  for (i in 1:nn) {
    x[i, 1] = x_init[i];
    z[i, 1] = z_init[i];
  } 
  
  for (t in 1:(nt-1)) {
    for (i in 1:nn) {
      gx = 0;
      for (j in 1:nn)
        gx = gx + SC[i, j]*(x[j, t] - x[i, t]);
      dx = 1.0 - x[i, t]*x[i, t]*x[i, t] - 2.0*x[i, t]*x[i, t] - z[i, t] + I1;
      dz = (1/tau0)*(4*(x[i, t] - x0[i]) - z[i, t] - K*gx);
      x[i, t+1] = x[i, t] + dt*dx ; 
      z[i, t+1] = z[i, t] + dt*dz ; 
    }
  }  
  
  
  for(i in 1:nt){
    xhat_qqc2[i,]=to_row_vector(amp * (gain * x[,i]) + u);
  }
  
  xhat_qqc=to_vector(xhat_qqc2);
  
  for (i in 1:(nt*ns)){
    x_ppc[i] = normal_rng(xhat_qqc[i], eps);
  }
  
  for (i in 1:(nt*ns)){
    log_lik[i] = normal_lpdf(xs[i]| xhat_qqc[i], eps);
  }
  
  
  for(i in 1:nt){
    xhat_q2[i,]=to_row_vector(amp * (gain * x[,i]) + u);
  }
  
  xhat_q=xhat_q2;
  
  for (i in 1:nt){
    for (j in 1:ns)
      x_p[i,j] = normal_rng(xhat_q[i,j], eps);
  }
  
  for (i in 1:nt){
    for (j in 1:ns)
      log_lik_sum[i] += normal_lpdf(Obs[i,j]| xhat_q[i,j], eps);
  }    
}
