functions {
   matrix correct_baseline(matrix Seeg, int n_bsl) {
        int nt = rows(Seeg);
        int ns = cols(Seeg);
        matrix[nt, ns] output;
        real bsl_value;
         if (n_bsl==0)
           return Seeg;
         for (s in 1:ns) {
         bsl_value = mean(sub_col(Seeg, 1, s, n_bsl));
         output[, s] = Seeg[, s] - bsl_value;
        }
       return output;
    }
   vector ode_rhs(real time, vector xz, matrix SC, real I1, real tau0, real K, vector eta) {
    int nn = rows(xz)%/%2;
    vector[nn] x = xz[1:nn];
    vector[nn] z = xz[nn+1:2*nn];
    vector[nn] gx = SC * x;
    // print(nn, " ", rows(x), " ", rows(z), " ", rows(eta), " ", rows(gx));
    vector[nn] dx = 1.0 - x.*x.*x - 2.0*x.*x - z + I1;
    vector[nn] dz = (1/tau0)*(4*(x - eta) - z - K*gx);
    return append_row(dx, dz);
  }


   vector ode_step(real time, real dt, vector xz, matrix SC, real I1, real tau0, real K, vector eta) {
    vector[rows(xz)] dxz = ode_rhs(time,xz,SC,I1,tau0,K,eta);
    //vector[rows(xz)] d2 = ode_rhs(time+dt, xz+dt*d1,SC,I1,tau0,K,eta);
    return xz + dt*dxz;
  //return xz + dt / 2 * (d1 + d2);
  }

//matrix ode_sol(real dt, int nt, vector xz, matrix SC, real I1, real tau0, real K, vector eta);
  matrix ode_sol(real dt, int nt, vector xz, matrix SC, real I1, real tau0, real K, vector eta) {
    matrix[rows(xz),nt] sol;
    sol[,1] = xz;
    for (t in 1:(nt - 1)) {
      sol[,t+1] = ode_step(t*dt, dt, sol[,t], SC, I1, tau0, K, eta);
    }
    return sol;
  }
  }
data {
  int nn;
  int nt;
  int ns1;
  int ns2;
  int nbs1;
  int nbs2;
  
  matrix[nt, ns1] Obs1;
  matrix[nt, ns2] Obs2;
  matrix [nn, nn] SC;
  matrix[ns1, nn] gain1;
  matrix[ns2, nn] gain2;
  matrix[nn,nn] eig1;
  matrix[nn,nn] eig2;
}

transformed data {
  real dt;
  real tau0;
  real I1;
  real Ks;
  
  vector[nt*ns1] xs1;
  vector[nt*ns2] xs2;
  matrix[nt,ns1] wtm1;
  vector[nt*ns1] wt1;
  matrix[nt,ns2] wtm2;
  vector[nt*ns2] wt2;
  vector[nn] x_init;
  for(i in 1:nn){
    x_init[i]=-1.98;
  }
  
  // xs=to_vector(log(Obs));
  xs1 = to_vector(Obs1);
  xs2 = to_vector(Obs2);

  for(i in 1:nt){
    wtm1[i,]=Obs1[i,:]-Obs1[1,:];
    wtm2[i,]=Obs2[i,:]-Obs2[1,:];
  }
  wt1=to_vector(wtm1);
  wt2=to_vector(wtm2);
  
  dt=0.1;
  tau0=15.0; // because i use 500 time points (10.0 for 180 time points)
  I1=3.1;
  Ks=1.0;
  
  
  
  
  
}

parameters {
  
  vector[nn] z_init_star;
  vector[nn] x0_star;
  real <lower=-1.0> K_star;
  real <lower=0.0> amp1_star;
  real <lower=0.0> amp2_star;
  vector[ns1] u1_star;
  vector[ns2] u2_star;
  real log_eps1_sq;
  real log_eps2_sq;
}

transformed parameters {
  
  vector[nn] x0;
  vector[nn] z_init;
  real K;
  real amp;
  real amp1;
  real amp2;
  vector[ns1] u1;
  vector[ns2] u2;
  real eps1; 
  real eps2;


  x0 = -3.0 + 0.30*eig1*x0_star;
  z_init=4.1+0.1*eig1*z_init_star;
  K =  Ks + 1.0*K_star;
  amp1 = 0.005 + 0.01*amp1_star;
  amp2 = 0.005 + 0.01*amp2_star;
  u1= 2.0+1.0*u1_star;
  u2= 2.0+1.0*u2_star;
  eps1=exp(0.33*log_eps1_sq-3.0); 
  eps2=exp(0.33*log_eps2_sq-3.0); 
  
  
}

model {
  matrix[nn, nt] x;
  matrix[nn, nt] z;
  matrix[nt, ns1] xhat2_1;
  vector[nt*ns1] xhat_1;
  matrix[nt, ns2] xhat2_2;
  vector[nt*ns2] xhat_2;
  matrix[nt,ns1] SEEG_bcor;
  vector[nt*ns1] SEEG;
  matrix[nt,ns2] EEG_bcor;
  vector[nt*ns2] EEG;

  real dx;
  real dz;
  real gx;
  
  // "group-level (hierarchical) SD across betas"
  
  x0_star ~ normal(0.,1.);  
  amp1_star ~ normal(0. ,1.);
  amp2_star ~ normal(0. ,1.);
  K_star ~ normal(0., 1.);  
  log_eps1_sq ~ normal(0.,10.0);
  log_eps2_sq ~ normal(0.,10.0);
  u1_star ~ normal(0., 1.0);
  u2_star ~ normal(0., 1.0);
  z_init_star ~ normal(0, 1.);
  
  /* integrate & predict */
    
//    for (i in 1:nn) {
  //    x[i, 1] =  x_init[i];
    //  z[i, 1] =  z_init[i];
    //} 
  
  //for (t in 1:(nt-1)) {
   // for (i in 1:nn) {
     // gx = 0;
      //for (j in 1:nn)
        //gx = gx + SC[i, j]*(x[j, t] - x[i, t]);
      //dx = 1.0 - x[i, t]*x[i, t]*x[i, t] - 2.0*x[i, t]*x[i, t] - z[i, t] + I1;
      //dz = (1/tau0)*(4*(x[i, t] - x0[i]) - z[i, t] - K*gx);
      //x[i, t+1] = x[i, t] + dt*dx ; 
      //z[i, t+1] = z[i, t] + dt*dz ; 
    //}
  //}   
  
  matrix[2*nn,nt] sol = ode_sol(dt, nt, append_row(x_init, z_init), SC, I1, tau0, K, x0);
  x= sol[1:nn,];

  for(i in 1:nt){
    xhat2_1[i,]=to_row_vector(amp1 * (gain1 * x[,i]) +  u1);
    xhat2_2[i,]=to_row_vector(amp2 * (gain2 * x[,i]) +  u2);
  }
  
  // xhat=to_vector(xhat2);
  SEEG_bcor = correct_baseline(xhat2_1, nbs1);
  SEEG = to_vector(SEEG_bcor);
  EEG_bcor = correct_baseline(xhat2_2, nbs2);
  EEG = to_vector(EEG_bcor);

  for(i in (10*ns1+1):(nt*ns1)){
    target+=  normal_lpdf(xs1[i]| SEEG[i], eps1);  
  }
  for(i in (10*ns2+1):(nt*ns2)){
    target+=  normal_lpdf(xs2[i]| EEG[i], eps2);
   }
}    

generated quantities {
  
  matrix[nn, nt] x;
  matrix[nn, nt] z;
  matrix[nt, ns1] xhat_qqc2_1;
  matrix[nt,ns1] xhat_q2_1;
  matrix[nt, ns2] xhat_qqc2_2;
  matrix[nt,ns2] xhat_q2_2;

  
  vector[nt*ns1] xhat_qqc_1;
  vector[nt*ns1] x_ppc_1;
  vector[nt*ns1] log_lik_1;
  
  matrix[nt,ns1] xhat_q_1;
  matrix[nt,ns1] x_p_1;
  vector[nt] log_lik_sum = rep_vector(0,nt);
  
  vector[nt*ns2] xhat_qqc_2;
  vector[nt*ns2] x_ppc_2;
  vector[nt*ns2] log_lik_2;
  matrix[nt,ns2] xhat_q_2;
  matrix[nt,ns2] x_p_2;
  
  matrix[nt,ns1] SEEG_bcor;
  vector[nt*ns1] SEEG;

  matrix[nt,ns2] EEG_bcor;
  vector[nt*ns2] EEG;

  real gx;
  real dx;
  real dz;
  
  int num_params;
  int num_data;
  num_params=2*(nt*(ns1+ns2))+nn+6;
  num_data=nt*(ns1+ns2);
  
  
 // for (i in 1:nn) {
   // x[i, 1] = x_init[i];
    //z[i, 1] = z_init[i];
 // } 
  
  //for (t in 1:(nt-1)) {
    //for (i in 1:nn) {
      //gx = 0;
      //for (j in 1:nn)
        //gx = gx + SC[i, j]*(x[j, t] - x[i, t]);
      //dx = 1.0 - x[i, t]*x[i, t]*x[i, t] - 2.0*x[i, t]*x[i, t] - z[i, t] + I1;
      //dz = (1/tau0)*(4*(x[i, t] - x0[i]) - z[i, t] - K*gx);
      //x[i, t+1] = x[i, t] + dt*dx ; 
      //z[i, t+1] = z[i, t] + dt*dz ; 
    //}
  //
  //}
  matrix[2*nn,nt] sol = ode_sol(dt, nt, append_row(x_init, z_init), SC, I1, tau0, K, x0);
  x = sol[1:nn,];
  z = sol[nn+1:,];
  
  for(i in 1:nt){
    xhat_qqc2_1[i,]=to_row_vector(amp1 * (gain1 * x[,i]) +  u1);
    xhat_qqc2_2[i,]=to_row_vector(amp2 * (gain2 * x[,i]) +  u2);
   }
   
   xhat_qqc_1=to_vector(xhat_qqc2_1);
   xhat_qqc_2=to_vector(xhat_qqc2_2);
   // xhat=to_vector(xhat2);
    SEEG_bcor = correct_baseline(xhat_qqc2_1, nbs1);
    SEEG = to_vector(SEEG_bcor);
    EEG_bcor = correct_baseline(xhat_qqc2_2, nbs2);
    EEG = to_vector(EEG_bcor);

  
  for (i in 1:(nt*ns1)){
    x_ppc_1[i] = normal_rng(xhat_qqc_1[i], eps1);
  }
  
  for (i in 1:(nt*ns1)){
    log_lik_1[i] = normal_lpdf(xs1[i]| xhat_qqc_1[i], eps1);
  }
  
  
  for(i in 1:nt){
    xhat_q2_1[i,]=to_row_vector(amp1 * (gain1 * x[,i]) + u1);
  }
  
  //xhat_q=xhat_q2;
  
  for (i in 1:nt){
    for (j in 1:ns1)
      x_p_1[i,j] = normal_rng(xhat_q2_1[i,j], eps1);
  }
  





  for (i in 1:(nt*ns2)){
     x_ppc_2[i] = normal_rng(xhat_qqc_2[i], eps2);
   }
 
   for (i in 1:(nt*ns2)){
     log_lik_2[i] = normal_lpdf(xs2[i]| xhat_qqc_2[i], eps2);
  }
 
 
   for(i in 1:nt){
     xhat_q2_2[i,]=to_row_vector(amp2 * (gain2 * x[,i]) + u2);
   }
 
   //xhat_q=xhat_q2;
 
   for (i in 1:nt){
     for (j in 1:ns2)
       x_p_2[i,j] = normal_rng(xhat_q2_2[i,j], eps2);
   }




  for (i in 1:nt){
    for (j in 1:ns1)
      log_lik_sum[i] += normal_lpdf(Obs1[i,j]| xhat_q2_1[i,j], eps1);
    for (k in 1:ns2)
      log_lik_sum[i] += normal_lpdf(Obs2[i,k]| xhat_q2_2[i,k], eps1);
  }    
}
