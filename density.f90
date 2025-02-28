module clus
    implicit none
    private gauss, outer_product, image_displ

contains

function gauss(dx, det, icov)
    real*8, intent(in) :: dx(:), det, icov(:,:)
    real*8 :: gauss, r2

    r2 = sum(dx * matmul(icov, dx))
    gauss = 1 / det * exp(-0.5d0 * r2)

end function gauss

function outer_product(a, b, n) result(prod)
    integer*4, intent(in) :: n
    real*8, intent(in) :: a(n), b(n)
    real*8 :: prod(n,n)
    prod = spread(source = a, dim = 2, ncopies = n) * spread(source = b, dim = 1, ncopies = n)
end function outer_product

subroutine mlloof(cov, nit, pts, pbc, thresh, nd, nt)
    implicit none

    ! external dsymv
    ! real*8, external :: ddot

    integer*4, intent(in) :: nd, nt, pbc(nd)
    integer*4, intent(out) :: nit
    real*8, intent(in) :: pts(nd, nt), thresh
    real*8, intent(inout) :: cov(nd, nd)


    integer*4 :: i, j
    real*8 tot(nd, nd), num(nd, nd), den, disp(nd)
    real*8 out(nd, nd, nt, nt), gs(nt, nt), det, icov(nd, nd), err, r2

    nit = 0
    do while(.true.)
        nit = nit + 1
        call det_and_inv(det, icov, cov, nd)
        do i = 1, nt
            do j = i+1, nt
                call image_displ(disp, pts(:,i), pts(:,j), pbc, nd)
                out(:,:,j,i) = outer_product(disp, disp, nd)
                out(:,:,i,j) = out(:,:,j,i)
                r2 = sum(disp * matmul(icov, disp))
                ! call dsymv('U', nd, 1d0, icov, nd, disp, 1, 0d0, temp, 1)
                ! r2 = sum(disp * temp)
                ! r2 = ddot(nd, disp, 1, temp, 1)
                gs(j,i) = dexp(-0.5d0 * r2)
                gs(i,j) = gs(j,i)
            end do
        end do

        tot = 0
        do i = 1, nt
            num = 0
            den = 0
            do j = 1, nt
                if (i == j) cycle
                num = num + out(:,:,j,i) * gs(j,i)
                den = den + gs(j,i)
            end do

            if (den > 1d-10) then
                tot = tot + num / den
            end if
        end do

        tot = tot / nt
        err = sum(abs(cov - tot))
        cov = tot
        ! print*, cov
        if (err < thresh) exit
    end do
end subroutine mlloof

subroutine mlloos(cov, nit, pts, pbc, thresh, nd, nt)
    implicit none

    ! external dsymv
    ! real*8, external :: ddot

    integer*4, intent(in) :: nd, nt, pbc(nd)
    integer*4, intent(out) :: nit
    real*8, intent(in) :: pts(nd, nt), thresh
    real*8, intent(inout) :: cov(nd, nd)


    integer*4 :: i, j
    real*8 tot, num, den, disp(nd), var
    real*8 dist(nt, nt), gs(nt, nt), err, r2

    var = 1
    nit = 0
    do while(.true.)
        nit = nit + 1
        do i = 1, nt
            do j = i+1, nt
                call image_displ(disp, pts(:,i), pts(:,j), pbc, nd)
                dist(j,i) = sum(disp * disp)
                dist(i,j) = dist(j,i)
                r2 = sum(disp * disp / var)
                ! call dsymv('U', nd, 1d0, icov, nd, disp, 1, 0d0, temp, 1)
                ! r2 = sum(disp * temp)
                ! r2 = ddot(nd, disp, 1, temp, 1)
                gs(j,i) = dexp(-0.5d0 * r2)
                gs(i,j) = gs(j,i)
            end do
        end do

        tot = 0
        do i = 1, nt
            num = 0
            den = 0
            do j = 1, nt
                if (i == j) cycle
                num = num + dist(j,i) * gs(j,i)
                den = den + gs(j,i)
            end do

            if (den > 1d-12) then
                tot = tot + num / den
            end if
        end do

        tot = tot / nt / nd
        err = abs(var - tot)
        var = tot
        print*, var
        if (err < thresh) then
            do i = 1, nd
                cov(i,i) = var
            end do
            exit
        end if
    end do
end subroutine mlloos

subroutine det_and_inv(det, inv, mat, n)
    implicit none

    external dsytri, dsytrf

    integer*4, intent(in) :: n
    real*8, intent(out) :: det
    real*8, intent(out) :: inv(n,n)
    real*8, intent(in) :: mat(n,n)

    integer*4 :: i, j, lda, info, lwork
    integer*4 :: ipiv(n)
    real*8 :: work(n)

    lda = n
    lwork = n

    inv(:,:) = mat

    call DSYTRF('U', n, inv, lda, ipiv, work, lwork, info)
    if (info /= 0) then
        print *, 'DSYTRF failed, INFO = ', info
        stop
    end if

    det = 1
    do i = 1, n
        det = det * inv(i,i)
    end do

    call DSYTRI('U', n, inv, lda, ipiv, work, info)
    if (info /= 0) then
        print *, 'DSYTRI failed, INFO = ', info
        stop
    end if

    do i = 1, n
        do j = i+1, n
            inv(j,i) = inv(i,j)
        end do
    end do
end subroutine det_and_inv

subroutine image_displ(displ, coords, source, pbc, n_features)
    integer*4, intent(in) :: n_features
    real*8, intent(in) :: coords(n_features), source(n_features)
    integer*4, intent(in) :: pbc(n_features)
    real*8, intent(out) :: displ(n_features)

    integer n

    do n = 1, n_features
        displ(n) = coords(n) - source(n)
        if (pbc(n) > 0) then
            if (displ(n) > 0.5) displ(n) = displ(n) - 1
            if (displ(n) < -0.5) displ(n) = displ(n) + 1
        end if
    end do
end subroutine image_displ

subroutine get_density_from_traj(density, displ, icov, det, coeff, n_features)
    implicit none

    real*8, intent(inout) :: density
!f2py intent(in,out) :: density

    integer*4, intent(in) :: n_features
    real*8, intent(in) :: displ(n_features), icov(n_features, n_features), det
    real*8, intent(in) :: coeff

    real*8 r2

    r2 = sum(displ * matmul(icov, displ))
    ! if (r2 .lt. 1d-14) return
    density = density + dexp(-0.5d0*coeff*r2)/sqrt(abs(det))

end subroutine get_density_from_traj

subroutine get_density(density, displs, icovs, dets, coeff, n_traj, n_features)
    implicit none

    real*8, intent(inout) :: density
!f2py intent(in,out) :: density

    integer*4, intent(in) :: n_traj, n_features
    real*8, intent(in) :: displs(n_features, n_traj), icovs(n_features, n_features, n_traj), dets(n_traj)
    real*8, intent(in) :: coeff

    integer i
    do i = 1, n_traj
        call get_density_from_traj(density, displs(:,i), icovs(:,:,i), dets(i), coeff, n_features)
    end do
end subroutine get_density

subroutine get_density_from_cluster(density, displs, icovs, dets, clusters, &
& coeff, clus_id, n_traj, n_features)
    implicit none

    real*8, intent(inout) :: density
!f2py intent(in,out) :: density

    integer*4, intent(in) :: n_traj, n_features, clus_id
    real*8, intent(in) :: displs(n_features, n_traj), icovs(n_features, n_features, n_traj), dets(n_traj)
    integer*4, intent(in) :: clusters(n_traj)
    real*8, intent(in) :: coeff

    integer i

    do i = 1, n_traj
        if (clusters(i) .eq. clus_id) &
        & call get_density_from_traj(density, displs(:,i), icovs(:,:,i), dets(i), coeff, n_features)
    end do
end subroutine get_density_from_cluster

subroutine get_density_by_clusters(contributions, displs, icovs, dets, clusters, &
& coeff, n_traj, n_features, n_clus)
    implicit none

    integer*4, intent(in) :: n_traj, n_features, n_clus
    real*8, intent(out) :: contributions(n_clus)
!f2py intent(out) :: contributions

    real*8, intent(in) :: displs(n_features, n_traj), icovs(n_features, n_features, n_traj), dets(n_traj)
    integer*4, intent(in) :: clusters(n_traj)
    real*8, intent(in) :: coeff

    integer i

    contributions = 0
    do i = 1, n_traj
        if (clusters(i) .ge. 0) &
        & call get_density_from_traj(contributions(clusters(i)+1), displs(:,i), icovs(:,:,i), dets(i), &
        &coeff, n_features)
    end do
end subroutine get_density_by_clusters

subroutine get_gradient(gradient, displs, icovs, dets, coeff, n_traj, n_features)
    implicit none

    integer*4, intent(in) :: n_traj, n_features

    real*8, intent(in) :: displs(n_features, n_traj), icovs(n_features, n_features, n_traj), dets(n_traj)
    real*8, intent(in) :: coeff

    real*8, intent(out) :: gradient(n_features)

    real*8 density
    integer i

    gradient = 0
    do i = 1, n_traj
        density = 0
        call get_density_from_traj(density, displs(:,i), icovs(:,:,i), dets(i), coeff, n_features)
        gradient = gradient - coeff*density*matmul(icovs(:,:,i), displs(:,i))
    end do
end subroutine get_gradient

subroutine get_hessian(hessian, displs, icovs, dets, coeff, n_traj, n_features)
    implicit none

    integer*4, intent(in) :: n_traj, n_features

    real*8, intent(in) :: displs(n_features, n_traj), icovs(n_features, n_features, n_traj), dets(n_traj)
    real*8, intent(in) :: coeff

    real*8, intent(out) :: hessian(n_features, n_features)

    real*8 density, temp(n_features)
    integer i, m, n

    hessian = 0
    do i = 1, n_traj
        density = 0
        call get_density_from_traj(density, displs(:,i), icovs(:,:,i), dets(i), coeff, n_features)
        temp = coeff*matmul(icovs(:,:,i), displs(:,i))
        do m = 1, n_features
            hessian(m,m) = hessian(m,m) + density*(temp(m)**2 - coeff*icovs(m,m,i))
            do n = 1, m-1
                hessian(n,m) = hessian(n,m) + density*(temp(n)*temp(m) - coeff*icovs(n,m,i))
                ! needs to be optimised
                hessian(m,n) = hessian(m,n) + density*(temp(n)*temp(m) - coeff*icovs(n,m,i))
            end do
        end do
    end do
end subroutine get_hessian

subroutine newton_raphson(xout, nit, xin, pts, icovs, dets, pbc, coeff, trust, tol, nt, nd)
    external :: dsyev
    integer*4, intent(in) :: nt, nd, pbc(nd)
    real*8, intent(in):: xin(nd), pts(nd, nt), icovs(nd, nd, nt), dets(nt), coeff, trust, tol
    real*8, intent(out) :: xout(nd)
    integer*4, intent(out) :: nit

    integer*4 i, lwork, info
    real*8 gk(nd), hk(nd, nd), displs(nd, nt), lam(nd), vec(nd, nd), dx(nd), gbar(nd), norm
    real*8, allocatable :: work(:)

    lwork = 3*nd - 1
    allocate(work(lwork))

    xout(:) = xin
    nit = 0

    do while(.true.)
        nit = nit + 1
        do i = 1, nt
            displs(:,i) = xout - pts(:,i)
            call image_displ(displs(:,i), xout, pts(:,i), pbc, nd)
        end do
        call get_gradient(gk, displs, icovs, dets, coeff, nt, nd)
        gk = -gk / nt
        call get_hessian(hk, displs, icovs, dets, coeff, nt, nd)
        hk = -hk / nt
        vec(:,:) = hk
        call dsyev('V', 'U', nd, vec, nd, lam, work, lwork, info)
        gbar = matmul(transpose(vec), gk)

        dx = 0
        do i = 1, nd
            dx = dx - gbar(i) / abs(lam(i)) * vec(:,i)
        end do

        norm = sqrt(sum(dx**2))
        if (norm < trust) then
            xout = xout + dx
        else
            xout = xout + dx / norm * trust
        end if

        if (norm < tol) return
    end  do

end subroutine newton_raphson

! subroutine eigtest()
!     external :: dsyev
!     integer*4 nd, info, lwork
!     real*8 mat(3,3), lam(3), vec(3,3)
!     real*8, allocatable :: work(:)

!     nd = 3
!     lwork = 3*nd - 1
!     allocate(work(lwork))

!     mat(:,1) = [1,2,3]
!     mat(:,2) = [2,4,2]
!     mat(:,3) = [3,2,0]
!     vec(:,:) = mat

!     call dsyev('V', 'U', nd, vec, nd, lam, work, lwork, info)
!     print*, vec(1,:)
!     print*, matmul(mat, vec(1,:))
!     print*, lam(1) * vec(1,:)
!     print*, ''
!     print*, vec(:,1)
!     print*, matmul(mat, vec(:,1))
!     print*, lam(1) * vec(:,1)



! end subroutine eigtest

end module clus