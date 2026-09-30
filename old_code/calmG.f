      PROGRAM calm_time
      implicit none
      integer maxstock,max_esc,max_bin
      parameter (maxstock=2000,max_esc=2000,max_bin=1000)
      integer i,j,day,time,numdata,l,pos,riga,numstock
      integer ncalm,cnt(maxstock),num_bin
      real*8 soglia,calm(maxstock),sstart,sstart2,sfinish,rit_anom
      real*8 price0(maxstock),price(maxstock),rit,sigma_max,delta
      real*8 avg(maxstock),sigma(maxstock),taumin,sigma_tot
      real*8 avg_loc(maxstock),sigma_loc(maxstock)
      real*8 isto_s(max_bin),isto_q(max_bin)
      integer isto_n(max_bin),ipos
      logical run(maxstock)
      character*100 progname,formato,infile,outfile,parmfile
      namelist /parm/sstart,sstart2,sfinish,numstock,rit_anom,taumin,
     +          sigma_max,num_bin
      
      !IF (iargc().ne.3) THEN
      !  call getarg(0,progname)
	!l=len_trim(progname)+1
	!IF (l.lt.10) WRITE(formato,"('(''Usage: '',a',i1,',''<infile> <outfile> <param file>'')')") l
	!IF (l.ge.10.and.l.lt.100) WRITE(formato,"('(''Usage: '',a',i2,',''<infile> <outfile> <param file>'')')") l
       ! WRITE(*,formato) progname
        !STOP
      !END IF
      !call getarg(1,infile)
      !call getarg(2,outfile)
      !call getarg(3,parmfile)

	 infile='dprice_dp_simul.dat' 
      !infile='dx_simul.dat'
	outfile='tau_mean_vs_noise_simul.dat'
	parmfile='parmG.dat'

      OPEN(10,file=parmfile)
      READ(10,nml=parm)
      CLOSE(10)
      delta=sigma_max/num_bin
      DO i=1,max_bin
        isto_s(i)=0.0
        isto_n(i)=0.0
        isto_q(i)=0.0
      END DO
      IF (numstock.gt.maxstock) THEN
        PRINT *,'Too much stock, max ',maxstock,'. Recompile program'
        STOPs
      END IF

      sigma_tot=0.0
	
      OPEN(10,file=infile)
      OPEN(20,file=outfile)
      DO i=1,numstock
        avg(i)=0
        sigma(i)=0
        avg_loc(i)=0
        sigma_loc(i)=0
        cnt(i)=0
      END DO
      riga=0
      DO
        READ(10,100,END=999) day,(price(i),i=1,numstock)
        riga=riga+1
        DO i=1,numstock
          rit=price(i)
          IF (abs(rit).gt.rit_anom) rit=0.0
          avg(i)=avg(i)+rit
          sigma(i)=sigma(i)+rit**2
        END DO  
      END DO
999   CLOSE(10)
      DO i=1,numstock
        avg(i)=avg(i)/riga
        sigma(i)=sqrt(sigma(i)/riga-avg(i)**2)
        PRINT *,'stock',i,'   avg =',avg(i), '   sigma =', sigma(i)
	  sigma_tot=sigma_tot+sigma(i)
      END DO

	sigma_tot=sigma_tot/numstock

	PRINT *,'sigma_tot =', sigma_tot

	! Scrive in un file la volatilit� per ogni serie temporale (stock option)
	
      ! **************** inizio ************************************
	
	    OPEN(15,file='volatility.dat',status='unknown')

        DO i=1,numstock

          WRITE(15,*)sigma(i)

	      END DO

      CLOSE(15)
	
  	  ! **************** fine ************************************



      DO i=1,maxstock
        run(i)=.FALSE.
        calm(i)=0.
      END DO
      numdata=0
      OPEN(10,file=infile)
      ! nuovo !
      OPEN(44,file='tau_simul.dat')
        OPEN(45,file='tau_vs_noise_simul.dat')
          riga=0
          DO
            READ(10,100,END=1000) day,(price(i),i=1,numstock)
            riga=riga+1
            IF (mod(riga,500).eq.0) 
              PRINT *,riga
            numdata=numdata+1
            DO i=1,numstock
              rit=price(i)
              IF (run(i)) THEN
                avg_loc(i)=avg_loc(i)+rit
                sigma_loc(i)=sigma_loc(i)+rit**2
                cnt(i)=cnt(i)+1
              END IF
              IF (abs(rit).gt.rit_anom) 
                rit=0.0
              ! IF (rit.gt.sstart*sigma(i)) run(i)=(.TRUE.)
              ! IF (rit.gt.sstart*sigma(i).and.rit.lt.sstart2*sigma(i)) !by Giovanni
              IF (rit.gt.sstart*sigma_tot.and.rit.lt.sstart2*sigma_tot) !by Davide
                1 run(i)=(.TRUE.)
              ! IF (rit.le.sfinish*sigma(i).and.run(i)) THEN  !by Giovanni
              IF (rit.le.sfinish*sigma_tot.and.run(i)) THEN  !by Davide
                run(i)=.FALSE.
                IF (calm(i).ge.taumin.and.calm(i).le.300) THEN
                  ! PRINT *,calm(i)
                  WRITE(44,*) calm(i)
                  avg_loc(i)=avg_loc(i)/cnt(i)
                  sigma_loc(i)=sigma_loc(i)/cnt(i)-avg_loc(i)**2
                  IF (sigma_loc(i).gt.0) THEN
                    sigma_loc(i)=sqrt(sigma_loc(i))
                    ! original
                    ! sigma_loc(i)=sqrt(sigma_loc(i)*cnt(i)/250)
                    ! Rosario Mantegna
                  ELSE 
                    sigma_loc(i)=0
                  END IF        
                  WRITE(45,*)sigma_loc(i), calm(i)	    ! nuovo (davide)
                  ipos=int(sigma_loc(i)/delta)+1
                  IF (ipos.le.max_bin) THEN
                    !PRINT *,ipos
                    isto_s(ipos)=isto_s(ipos)+calm(i)
                    isto_q(ipos)=isto_q(ipos)+calm(i)**2
                    isto_n(ipos)=isto_n(ipos)+1
                  END IF
                  !WRITE(20,'(3f15.8)') avg_loc(i),sigma_loc(i),calm(i)
                  cnt(i)=0
                  avg_loc(i)=0.0
                  sigma_loc(i)=0.0
                  calm(i)=0
                END IF  
              ELSE
                IF (run(i)) calm(i)=calm(i)+1
              END IF
            END DO
          END DO
      
1000  continue
      CLOSE(10)
      CLOSE(44)
      CLOSE(45)
      !CLOSE(20)
      OPEN(20,file=outfile)
      DO i=1,max_bin
        IF (isto_n(i).ne.0) THEN
          isto_s(i)=isto_s(i)/isto_n(i)
          isto_q(i)=sqrt(isto_q(i)/isto_n(i)-isto_s(i)**2)
          WRITE(20,'(f15.8,i8,2f15.8)')
     +      i*delta/2,isto_n(i),isto_s(i),isto_q(i)
        END IF
      END DO
      CLOSE(20)
100   format(i6,3x,2000f12.7)
200   format(1000(f10.4,'\t'))
      END
