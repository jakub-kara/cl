subroutine modri(a, p)
    real*8, intent(inout) :: a
    integer*4, intent(in) :: p

    do while (a > 0)
        a = a - p
    end do
    a = a + p

end subroutine modri

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

subroutine find_closest_points(closest_pts, pair_dist, path_vec, dataset, clusters, &
& icov, pbc, clus1, clus2, n_traj, n_features, n_pairs)
    implicit none

    integer*4, intent(in) :: clus1, clus2, n_traj, n_features, n_pairs, pbc(n_features)
    integer*4, intent(out) :: closest_pts(2, n_pairs)
!f2py intent(out) :: closest_pts
    real*8, intent(out) :: pair_dist(n_pairs), path_vec(n_features, n_pairs)
!f2py intent(out) :: pair_dist
    real*8, intent(in) :: dataset(n_features, n_traj), icov(n_features, n_features)
    integer*4, intent(in) :: clusters(n_traj)

    integer i, j, c, d
    real*8 r2, vec(n_features), displ(n_features)
    pair_dist = 1d307
    do i = 1, n_traj
        if (clusters(i) .ne. clus1) cycle

        do j = 1, n_traj
            if (clusters(j) .ne. clus2) cycle

            call image_displ(displ, dataset(:,i), dataset(:,j), pbc, n_features)
            r2 = sum(displ * matmul(icov, displ))
            vec = -displ/sqrt(r2)

            do c = 1, n_pairs
                if (r2 .lt. pair_dist(c)) then
                    do d = n_pairs, c+1, -1
                        pair_dist(d) = pair_dist(d-1)
                        closest_pts(:,d) = closest_pts(:,d-1)
                        path_vec(:,d) = path_vec(:,d-1)
                    end do
                    pair_dist(c) = r2
                    closest_pts(1,c) = i-1
                    closest_pts(2,c) = j-1
                    path_vec(:,c) = vec
                    exit
                end if
            end do
        end do
    end do
end subroutine find_closest_points

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
        gradient = gradient - density*matmul(icovs(:,:,i), displs(:,i))
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
        temp = matmul(icovs(:,:,i), displs(:,i))
        do m = 1, n_features
            hessian(m,m) = hessian(m,m) + density*(temp(m)**2 - icovs(m,m,i))
            do n = 1, m-1
                hessian(n,m) = hessian(n,m) + density*(temp(n)*temp(m) - icovs(n,m,i))
                ! needs to be optimised
                hessian(m,n) = hessian(m,n) + density*(temp(n)*temp(m) - icovs(n,m,i))
            end do
        end do
    end do
end subroutine get_hessian
!
!
!subroutine gradient_ascend(output, dataset, feature_vars, coeff, n_traj, n_features)
!    implicit none
!
!    integer*4, intent(in) :: n_traj, n_features
!    integer*4, intent(out) :: output(n_features, n_traj)
!!f2py intent(out) :: maxima
!
!    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_features)
!    real*8, intent(in) :: coeff
!
!    real*8 coords(n_features), temp_coords(n_features), gradient(n_features), tol
!    integer i, n, count
!
!
!    tol = 1d-10
!    do i = 1, n_traj
!        coords = dataset(:,i)
!        do while (.true.)
!            gradient = 0
!            call get_gradient(gradient, dataset, feature_vars, coords, coeff, n_traj, n_features)
!
!            if (norm2(gradient) < tol) exit
!
!            coords = 0
!        end do
!    end do
!
!end subroutine gradient_ascend