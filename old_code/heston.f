      program cubico
      implicit none
      integer tau_max,nbin,vistomax,num_a,k_a, number, count, ctrl
	  integer j_exp,i_time,j_int,numstock,time_step
      parameter (tau_max=1000,nbin=100,vistomax=100000,numstock=1071)	!1071
	parameter(time_step=3030)
      integer i,j,evt,try,tau,numeventi,nexp_2,indexisto,ind_isto,numv
      integer isto(tau_max),nd,npt,num,reply,visto(vistomax)

      !real*8 W,price,soglia,stau,D,Dmin,Dmax,start,mu,dU_dx_U_0
      !real*8 a,b,t,dt,x,dx,dx2(1075,3035),dU_dx,sum,kc,q,coda,factor,t2
      !real*8 Vstart,dlogp,V,V_new,aa,bb,cc,alpha
	!real*8 sump, sump2, sdev, start2, soglia2, sumv, sumv2
	!real*8 sstart2, sfinish2, j_real, j_real2
	!real*8 vmean, Vdev, Vmean_fin, dev_fin, Vdev_fin   

	double precision W,dprice_dt,soglia,stau,D,Dmin,Dmax,start,mu
	double precision W_c,rho,dZ
	double precision dU_dx_U_0, x_U_min, x_U_max, x_old, dp_abs
      double precision a,b,t,dt,x,dx,dU_dx,sum,kc,q,coda
	double precision dx_2(1072,3032),dx_2_square(1072,3032),V_2(1072,3032)
      double precision Vstart,dlogp,V,V_new,aa,bb,cc,alpha,factor,t2
	double precision sump, sump2, sdev, start2, soglia2, sumv, sumv2
	double precision sstart2, sfinish2, j_real, j_real2, soglia3
	double precision vmean, Vdev, Vmean_fin, dev_fin, Vdev_fin
	double precision sumdx,meandx,sumdx2,meandx2,sumdx3,meandx3
	double precision sumdx4,meandx4,sigmadx,mu3,mu4,skewdx,kurtdx 
	integer ndata

      character*5 si
	character*40 filename, filename_pdf

      namelist /parm/soglia,start,Vstart,dt,
     1               a,b,Dmin,Dmax,rho,numeventi,nexp_2,
     2               aa,bb,cc,npt,factor,sstart2,sfinish2

      
!      open(60,file='values_of_a.dat',status='unknown')
!	  read(60,*) num_a

!	open(20,file='namelist_pdf.dat',status='unknown')
		    	       
!      do k_a = 1, num_a

!	  print *,'  step', k_a, ' of', num_a 
!        read(60,*) aa, filename
	  	
     
      nd=0
      !open(30,file='tau_mean.dat')
      open(10,file='parm.dat')
      read(10,nml=parm)
      close(10)

	! nuovo nuovo
	! b=(0.4*a**2)**0.33
	!start=-0.66*(b/a)-0.1*0.66*(b/a)
	!dU_dx_U_0 = (0.4)**0.66*a**0.33 
	! fine nuovo nuovo
		
	! nuovo 
		
	!open(20,file=namelist_pdf,status='old')
  
	 ! cc=-aa*(bb-0.04)/sqrt(0.04)

   	  print *,'  aa =', aa, '  bb =', bb, '  cc =', cc, '  rho =', rho
	  print *,'   a =',  a, '   b =',  b
	  x_U_max = (-2*b)/(3*a)
	  x_U_min = 0
	  print *,'start =', start
	  print *,'minimo in x =', x_U_min
	  print *,'massimo in x =', x_U_max

	
	!  read(5,*) ctrl

      D=Dmin
      num=1
150   continue
      stau=0.0
      call randomize()
      do i=1,tau_max
        isto(i)=0.0
      end do
      numv=0
      evt=1
      try=0
      V=Vstart
      do i=1,vistomax
        visto(i)=0
      end do
      !cc=cc*32.0d0 !AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

   	  !nuovo
 !	  read(20,*) filename_pdf
      !open(25,file='pdf.dat')
 
    	!!!!!!!!!!!!!!!!!!!

   	   !!!!!!!!!!!!!!!!!!!!!!!! nuovo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 	Vmean_fin = 0.0
	 dev_fin = 0.0 
	 Vdev_fin = 0.0 
 
       do i = 1, nexp_2
 	  x=start
        !V=Vstart
	  count=0
        t=0.0
	  sump2=0.0
	  sump=0.0
	  sumv2=0.0
	  sumv=0.0
	  Vmean=0.0    
	  Vdev=0.0     
	  sdev=0.0
	   do while (x.gt.soglia.and.t.le.real(tau_max))
	   count=count+1
          t=t+1
          x_old=x
      !    dprice_dt=x-dU_dx(a,b,x,t)*dt+sqrt(V)*W(0.0d0,0.1d0)   !old
	 !    V=0.0
	    x=x_old-dU_dx(a,b,x,t)*dt-(V/2)*dt+dsqrt(V*dt)*W(0.0d0,1.0d0)   !new
	 !  x=x_old-dU_dx(a,b,x,t)*dt+dsqrt(V*dt)*W(0.0d0,1.0d0)	!new without Ito
	 !  dprice_dt=x-dU_dx(a,b,x,t)*dt+dsqrt(V*dt)*W(0.0d0,1.0d0)  !new new
	    dx=x-x_old

	   !    write(6,*)t, dx

		sump=sump+dx
		sump2=sump2+dx**2

          V_new=-1.0
          reply=0.0
          do while (V_new.lt.0)
            reply=reply+1
            if (reply.gt.500) then
              print *,'Bad parameters'
              stop
            end if
            V_new=V+aa*(bb-V)*dt+cc*sqrt(V*dt)*W(0.0d0,1.0d0)   !Heston old
	!      dZ=W(0.0d0,1.0d0)
	!      W_c=rho*dZ+sqrt(1-rho**2)*W(0.0d0,1.0d0)
	!      V_new=V+aa*(bb-V)*dt+cc*dsqrt(V*dt)*W_c   !Heston new

          end do 
          V=V_new 

	 !   x=x_old-dU_dx(a,b,x,t)*dt-(V/2)*dt+dsqrt(V*dt)*dZ
	!	dx=x-x_old 

	    sumv=sumv+V
		sumv2=sumv2+V**2

          ind_isto=indexisto(V,0.0d0,5*bb,nbin)
          if (ind_isto.lt.vistomax) then
            numv=numv+1
            visto(ind_isto)=visto(ind_isto)+1
          else 
            !print *,ind_isto,V
            !stop
          end if  
      !    price=price*exp(dlogp)
		                   
        end do

	     sump2=sump2/count
	     sump=sump/count

		 sdev=sqrt(sump2-sump**2)
	     
	     sumv2=sumv2/count
	     sumv = sumv/count

	     Vmean=sumv		        
		 Vdev=sqrt(sumv2-sumv**2)
		  
	     dev_fin = dev_fin + sdev
	     Vmean_fin = Vmean_fin + Vmean
	     Vdev_fin = Vdev_fin + Vdev


	!     write(6,*)'sdev =', sdev
	!     write(6,*)'Vmean =', Vmean
	!     write(6,*)'Vdev =', Vdev
		 
		 end do 

	     dev_fin = dev_fin/nexp_2
	     Vmean_fin = Vmean_fin/nexp_2	   
	     Vdev_fin = Vdev_fin/nexp_2
         
	     write(6,*)'sstart2 =', sstart2
	     write(6,*)'sfinish2 =', sfinish2

		 write(6,*)'dev_fin =', dev_fin
	     write(6,*)'Vmean_fin =', Vmean_fin
	     write(6,*)'Vdev_fin =', Vdev_fin

	     start2=sstart2*dev_fin
	     soglia2=sfinish2*dev_fin

	     soglia3=100*soglia2

	     !read(5,*) ctrl


	 !   write(6,*)count
	 !	 write(6,*)sump
	 !   write(6,*)sump2
	 !	 write(6,*)sdev
	 !   write(6,*)soglia2
	 
      
	   !!!!!!!!!!!!!!!!!!!!!!!! fine nuovo !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

	  j_real=0.0
		
        do j_exp = 1, numstock

	    j_real=j_real+1.0

	    j_real2 = j_real/500
	    j_int = int(j_real/500)

	   if(j_real2.eq.j_int) then 	   
	     write(6,*)'esperimento', j_exp
	   end if

        x=start
        V=Vstart
        t=0.0
		i_time=0
	    dx=0.0
	    t2=0.0
        !print *,'ciao',evt

	!!!!!!!!!!!!!!!!!!!!!!!!!!! nuovo 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

   	
    !!!!!!!!!!!!!!!!!!!!!!!! fine nuovo 2 !!!!!!!!!!!!!!!!!!!!!!!!!!!!

	  do i_time = 1, time_step
          !   do while (dx.gt.soglia2.and.i_time.le.3031)
          t=t+1
	!	  i_time=i_time+1
          x_old=x
      !    dprice_dt=x-dU_dx(a,b,x,t)*dt+sqrt(V)*W(0.0d0,0.1d0)   !old
	   x=x_old-dU_dx(a,b,x,t)*dt-(V/2)*dt+dsqrt(V*dt)*W(0.0d0,1.0d0) !new
	!  x=x_old-dU_dx(a,b,x,t)*dt+dsqrt(V*dt)*W(0.0d0,1.0d0)  !new without Ito
	   dx=x-x_old
	    
		V_new=-1.0
          reply=0.0
          do while (V_new.lt.0)
            reply=reply+1
            if (reply.gt.500) then
              print *,'Bad parameters'
              stop
            end if
            V_new=V+aa*(bb-V)*dt+cc*sqrt(V*dt)*W(0.0d0,1.0d0)   !Heston old
	 !     dZ=W(0.0d0,1.0d0)
	 !     W_c=rho*dZ+sqrt(1-rho**2)*W(0.0d0,1.0d0)
	 !     V_new=V+aa*(bb-V)*dt+cc*dsqrt(V*dt)*W_c !Heston new
          end do
	!	x=x_old-dU_dx(a,b,x,t)*dt-(V/2)*dt+dsqrt(V*dt)*dZ !new
	!   x=x_old-dU_dx(a,b,x,t)*dt+dsqrt(V*dt)*W(0.0d0,1.0d0)  !new without Ito
	 !   dx=x-x_old
	    dp_abs=abs(dx*exp(x_old))

	 !     if(dp_abs.lt.0.01) then
	 !      dx=0.0
	 !     end if

	    dx_2(j_exp,i_time)=dx
	    dx_2_square(j_exp,i_time)=dx**2
	    V_2(j_exp,i_time)=sqrt(V)

	    if(x.lt.soglia) then
	       x=start     !soglia
	    end if

 
          V=V_new 
          ind_isto=indexisto(V,0.0d0,5*bb,nbin)
          if (ind_isto.lt.vistomax) then
            numv=numv+1
            visto(ind_isto)=visto(ind_isto)+1
          else 
            !print *,ind_isto,V
            !stop
          end if  
      !    price=price*exp(dlogp)
		                   
        end do
        tau=int(t)

        try=try+1
        if (tau.le.tau_max) then
          isto(tau)=isto(tau)+1
          stau=stau+tau
          evt=evt+1
          !if (mod(evt,5000).eq.0) print *,evt
        end if 
         
      end do
      if (evt.ne.0) then
        stau=stau/evt
      else
        stau=10000
      end if
      !q=coda(numv*1.d-2,0.0d0,5*bb,nbin,visto)
      do i=1,vistomax
        visto(i)=0
      end do
      numv=0

	! Blocco da modificare a seconda che si voglia tau in funzione di bb o di cc
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      !alpha=2*aa*bb/(cc**2)
      !write(30,*) bb,stau
      !write(30,*) cc,stau
      print *,num,bb,stau
      !print *,num,cc,stau
      D=D*2.0
      bb=bb*factor
      !cc=cc*factor
	!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

      num=num+1
	!close(25)
      if (num.le.npt) goto 150
!      !close(30)
		
!      end do  !k_a

!	close(60)
!	close(20)


	    do j_exp=1,numstock
	      
	      dx_2(j_exp,1)=0.0
	      dx_2_square(j_exp,1)=0.0

		end do !j


      open(65,file='dprice_dp_simul.dat')
		
		do i_time=1,time_step
	      
	      write(65,'(i6,3x,1500f12.8)') 
     1       051117,(dx_2(j_exp,i_time),j_exp=1,numstock)
	     
	    end do !i

	close(65)

		
	open(75,file='pdf_dprice_dp_simul.dat')

		do j_exp=1,numstock
		 do i_time=1,time_step		
    
	      write(75,*) dx_2(j_exp,i_time),dx_2_square(j_exp,i_time),V_2(j_exp,i_time)
	
	     end do !i					
		end do !j
	
	  close(75)

	open(95,file='pdf_log_abs_dprice_dp_simul.dat')

		do j_exp=1,numstock		          ! nuovo per calcolo dei log abs dei ritorni
		 do i_time=1,time_step		
    
	     write(95,*) dx_2(j_exp,i_time), Log10(abs(dx_2(j_exp,i_time)+0.0000000001))
	
	     end do !i					
		end do !j
	
	  close(95)



	! calcolo di media, dev std, skewness, kurtosis

	  sumdx=0
	  sumdx2=0
	  sumdx3=0
	  sumdx4=0

	  open(75,file='pdf_dprice_dp_simul.dat')

		do j_exp=1,numstock
		 do i_time=1,time_step
	      
	       read(75,*) dx

		sumdx=sumdx+dx
		sumdx2=sumdx2+dx**2
		sumdx3=sumdx3+dx**3
		sumdx4=sumdx4+dx**4			  
	   
	     end do !i
		end do !j
	
	    close(75)

		ndata=numstock*time_step

		meandx=sumdx/ndata
		meandx2=sumdx2/ndata
		meandx3=sumdx3/ndata
		meandx4=sumdx4/ndata	   
						
		sigmadx=sqrt(meandx2-meandx**2)
		mu3=(meandx3+2*meandx**3-3*meandx2*meandx)/sigmadx**3
		mu4=(meandx4-6*meandx**4+12*meandx2*meandx**2-4*meandx*meandx3-3*meandx2**2)/sigmadx**4

		skewdx=mu3   !**0.33
		kurtdx=mu4   !**0.25 

		open(85,file='moments.dat')
		  write(85,*)'mean','          sigma','       skewness','     kurtosis'
          write(85,*)meandx,sigmadx,skewdx,kurtdx
		close(85)   

	  ! *******************************************


	  stop
      end
      
      integer function indexisto(v,min,max,nbin)
      implicit none
      real*8 v,min,max
      integer nbin
      
      indexisto=int(nbin*v/(max-min))+1
      return
      end
       
      real*8 function dU_dx(a,b,x,t)
      implicit none
      double precision a,b,x,t 
	!real*8 a,b,x,t
      dU_dx=3*a*x**2+2*b*x
      return
      end
      
      subroutine randomize()
      implicit none
	double precision ran2,r
      !real*8 ran2,r
      integer*4 seed
      
      open(10,file='/dev/urandom',RECL=4,FORM='UNFORMATTED')
      read(10,REC=1) seed
      close(10)
      seed=-abs(seed)
      r=ran2(seed)
      return
      end
      
      real*8 function W(media,sigma)
      implicit none
      double precision r1,r2,u,media,sigma,pi,ran2
	!real*8 r1,r2,u,media,sigma,pi,ran2
      integer seed
      
      r1=0
      seed=1
      do while (r1.eq.0.0)
        r1=ran2(seed)
      end do  
      r2=ran2(seed)
      pi=4*atan(1.0)
      u=dsqrt(-2*dlog(r1))*dcos(2.0*pi*r2)
      W=media+sigma*u
      return
      end
      
      real*8 function ran2(idum)
      implicit none
      integer idum,im1,im2,imm1,ia1,ia2,iq1,iq2,ir1,ir2,ntab,ndiv
      real*8 am,eps,rnmx
      parameter (im1=2147483563,im2=2147483399,am=(1.0/im1),imm1=im1-1)
      parameter (ia1=40014,ia2=40692,iq1=53668,iq2=52774,ir1=12211)
      parameter (ir2=3791,ntab=32,ndiv=1+(imm1/ntab),eps=1.2e-7)
      parameter (rnmx=1.0-eps)
      integer idum2,j,k,iv(ntab),iy
      save iv,iy,idum2
      data idum2/123456789/,iv/ntab*0/,iy/0/
      
      
      if (idum.le.0) then
        idum=max(-idum,1)
        idum2=idum
        do j=ntab+8,1,-1
          k=idum/iq1
          idum=ia1*(idum-k*iq1)-k*ir1
          if (idum.lt.0) idum=idum+im1
          if (j.le.ntab) iv(j)=idum
        end do
        iy=iv(1)
      end if
      k=idum/iq1
      idum=ia1*(idum-k*iq1)-k*ir1
      if (idum.lt.0) idum=idum+im1
      k=idum2/iq2
      idum2=ia2*(idum2-k*iq2)-k*ir2
      if (idum2.lt.0) idum2=idum2+im2
      j=1+iy/ndiv
      iy=iv(j)-idum2
      iv(j)=idum
      if (iy.lt.1) iy=iy+imm1
      ran2=min((am*iy),rnmx)
      return
      end
       
      real*8 function coda(cd,min,max,nbin,visto)
      implicit none
      real*8 cd,min,max
      integer a,b,nbin,visto(*)
      
      a=1
      b=2
      do while (.not.(visto(a).gt.cd.and.visto(b).lt.cd))
        a=a+1
        b=b+1
      end do
      coda=min+(real(a+b)/2  )*((max-min)/nbin)
      return
      end
      
