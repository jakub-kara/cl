subroutine modri(a, p)
    real*8, intent(inout) :: a
    integer*4, intent(in) :: p

    do while (a > 0)
        a = a - p
    end do
    a = a + p
    
end subroutine modri

subroutine get_density_from_traj(density, dataset, feature_vars, coords, coeff, pbc, n_features, n_pts)
    implicit none
    
    real*8, intent(inout) :: density
!f2py intent(in,out) :: density

    integer*4, intent(in) :: n_features, n_pts, pbc(n_features)
    real*8, intent(in) :: dataset(n_features), feature_vars(n_features), coords(n_features)
    real*8, intent(in) :: coeff    

    real*8 r2, temp, ocoords(n_features)
    integer m, n
    
    r2 = 0
    do n = 1, n_features
        temp = coords(n) - dataset(n)
        r2 = r2 + temp*temp/feature_vars(n)
    end do
    density = density + dexp(-coeff*r2)

    do m = 1, n_features
        if (pbc(m) > 0) then
            ocoords = coords
            ocoords(m) = ocoords(m) + n_pts
            r2 = 0
            do n = 1, n_features
                temp = ocoords(n) - dataset(n)
                r2 = r2 + temp*temp/feature_vars(n)
            end do
            density = density + dexp(-coeff*r2)

            ocoords(m) = ocoords(m) -2*n_pts
            r2 = 0
            do n = 1, n_features
                temp = ocoords(n) - dataset(n)
                r2 = r2 + temp*temp/feature_vars(n)
            end do
            density = density + dexp(-coeff*r2)
        end if
    end do
end subroutine get_density_from_traj

subroutine get_density(density, dataset, feature_vars, coords, coeff, pbc, n_traj, n_features, n_pts)
    implicit none
    
    real*8, intent(inout) :: density
!f2py intent(in,out) :: density

    integer*4, intent(in) :: n_traj, n_features, n_pts, pbc(n_features)
    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_features), coords(n_features)
    real*8, intent(in) :: coeff

    integer i
    do i = 1, n_traj
        call get_density_from_traj(density, dataset(:,i), feature_vars, coords, coeff, pbc, n_features, n_pts)
    end do
end subroutine get_density

subroutine get_density_from_cluster(density, dataset, feature_vars, coords, clusters, &
& coeff, pbc, clus_id, n_traj, n_features, n_pts)
    implicit none

    real*8, intent(inout) :: density
!f2py intent(in,out) :: density
    
    integer*4, intent(in) :: n_traj, n_features, clus_id, n_pts, pbc(n_features)
    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_features), coords(n_features)
    integer*4, intent(in) :: clusters(n_traj)
    real*8, intent(in) :: coeff

    integer i

    do i = 1, n_traj
        if (clusters(i) .eq. clus_id) &
        & call get_density_from_traj(density, dataset(:,i), feature_vars, coords, coeff, pbc, n_features, n_pts)
    end do
end subroutine get_density_from_cluster

subroutine get_density_by_clusters(contributions, dataset, feature_vars, coords, clusters, &
& coeff, pbc, n_traj, n_features, n_clus, n_pts)
    implicit none
    
    integer*4, intent(in) :: n_traj, n_features, n_clus, n_pts, pbc(n_features)
    real*8, intent(out) :: contributions(n_clus)
!f2py intent(out) :: contributions

    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_features), coords(n_features)
    integer*4, intent(in) :: clusters(n_traj)
    real*8, intent(in) :: coeff

    integer i

    contributions = 0
    do i = 1, n_traj
        if (clusters(i) .ge. 0) &
        & call get_density_from_traj(contributions(clusters(i)+1), dataset(:,i), feature_vars, coords, &
        &coeff, pbc, n_features, n_pts)
    end do
end subroutine get_density_by_clusters

subroutine get_contributions_at_datapoints(contributions, dataset, feature_vars, clusters, &
& coeff, pbc, n_traj, n_features, n_clus, n_pts)
    implicit none
    
    integer*4, intent(in) :: n_traj, n_features, n_clus, n_pts, pbc(n_features)
    real*8, intent(out) :: contributions(n_clus, n_traj)
!f2py intent(out) :: contributions

    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_features)
    integer*4, intent(in) :: clusters(n_traj)
    real*8, intent(in) :: coeff

    integer i
    do i = 1, n_traj

        call get_density_by_clusters(contributions(:,i), dataset, feature_vars, dataset(:,i), clusters, &
        & coeff, pbc, n_traj, n_features, n_clus, n_pts)

        contributions(clusters(i)+1, i) = contributions(clusters(i)+1, i) - 1
    end do 
end subroutine get_contributions_at_datapoints

subroutine find_closest_points(closest_pts, pair_dist, path_vec, dataset, clusters, &
& pbc, clus1, clus2, n_traj, n_features, n_pairs, n_pts)
    implicit none
    
    integer*4, intent(in) :: clus1, clus2, n_traj, n_features, n_pairs, n_pts, pbc(n_features)
    integer*4, intent(out) :: closest_pts(2, n_pairs)
!f2py intent(out) :: closest_pts
    real*8, intent(out) :: pair_dist(n_pairs), path_vec(n_features, n_pairs)
!f2py intent(out) :: pair_dist
    real*8, intent(in) :: dataset(n_features, n_traj)
    integer*4, intent(in) :: clusters(n_traj)

    integer i, j, c, d, n
    real*8 r2, temp, vec(n_features), ocoords(n_features)
    pair_dist = 1d307
    do i = 1, n_traj
        if (clusters(i) .ne. clus1) cycle
        
        do j = 1, n_traj
            if (clusters(j) .ne. clus2) cycle

            r2 = sum((nint(dataset(:,i)) - nint(dataset(:,j)))**2)
            vec = (nint(dataset(:,j)) - nint(dataset(:,i)))/sqrt(r2)
            do n = 1, n_features
                if (pbc(n) > 0) then
                    ocoords = dataset(:,i)
                    ocoords(n) = ocoords(n) + n_pts
                    temp = sum((ocoords - dataset(:,j))**2)
                    if (temp < r2) then
                        r2 = temp
                        vec = (dataset(:,j) - ocoords)/sqrt(r2)
                    end if

                    ocoords(n) = ocoords(n) - 2*n_pts
                    temp = sum((ocoords - dataset(:,j))**2)
                    if (temp < r2) then
                        r2 = temp
                        vec = (dataset(:,j) - ocoords)/sqrt(r2)
                    end if
                end if
            end do
            
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

subroutine distance_from_line(distance, point, linepoint, linevec, n_features)
    implicit none
    
    real*8, intent(out) :: distance
!f2py intent(out) :: distance

    integer*4, intent(in) :: n_features
    real*8, intent(in) :: point(n_features), linepoint(n_features), linevec(n_features)

    real*8 t
    t = sum((point - linepoint)*linevec)
    distance = sum((point - linepoint - t*linevec)**2)
    distance = sqrt(distance)
end subroutine distance_from_line

subroutine compute_merge_metric(metric, dataset, feature_vars, clusters, closest_pts, path_vec, &
& coeff, pbc, clus1, clus2, n_traj, n_features, n_pairs, n_clus, n_pts)
    implicit none
    
    real*8, intent(out) :: metric
!f2py intent(out) metric

    integer*4, intent(in) :: clus1, clus2, n_traj, n_features, n_pairs, n_clus, n_pts, pbc(n_features)
    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_features), path_vec(n_features, n_pairs)
    integer*4, intent(in) :: clusters(n_traj), closest_pts(2, n_pairs)
    real*8, intent(in) :: coeff
    
    real*8 coords1(n_features), coords2(n_features), ocoords(n_features), temp_coords(n_features), &
    &r2, density1, density2, density_mid, dist, temp
    real*8 path_coords(n_features)
    real*8 contributions(n_clus)
    integer path_dir(n_features)
    logical at_coords2, use1
    integer j, k, n, count

    metric = n_traj
    pair_loop: do k = 1, n_pairs
        do n = 1, n_features
            coords1(n) = nint(dataset(n,closest_pts(1,k)+1))
            coords2(n) = nint(dataset(n,closest_pts(2,k)+1))
        end do

        r2 = sum((coords1 - coords2)**2)
        if (r2 .lt. n_features) then
            metric = 0
            return
        do n = 1, n_features
            if (pbc(n) > 0) then
                ocoords = coords1
                ocoords(n) = ocoords(n) + n_pts
                r2 = sum((ocoords - coords2)**2)
                if (r2 .lt. n_features) then
                    metric = 0
                    return
                end if

                ocoords(n) = ocoords(n) - 2*n_pts
                r2 = sum((ocoords - coords2)**2)
                if (r2 .lt. n_features) then
                    metric = 0
                    return
                end if
            end if
        end do

        else
            density1 = 0
            call get_density_from_cluster(&
                &density1, dataset, feature_vars, coords1, clusters, coeff, pbc, clus1, n_traj, n_features, n_pts)
            call get_density_from_cluster(&
                &density1, dataset, feature_vars, coords1, clusters, coeff, pbc, clus1-n_traj, n_traj, n_features, n_pts)
            call get_density_from_cluster(&
                &density1, dataset, feature_vars, coords1, clusters, coeff, pbc, clus2, n_traj, n_features, n_pts)
            call get_density_from_cluster(&
                & density1, dataset, feature_vars, coords1, clusters, coeff, pbc, clus2-n_traj, n_traj, n_features, n_pts)

            density2 = 0
            call get_density_from_cluster(&
                &density2, dataset, feature_vars, coords2, clusters, coeff, pbc, clus1, n_traj, n_features, n_pts)
            call get_density_from_cluster(&
                &density2, dataset, feature_vars, coords2, clusters, coeff, pbc, clus1-n_traj, n_traj, n_features, n_pts)
            call get_density_from_cluster(&
                &density2, dataset, feature_vars, coords2, clusters, coeff, pbc, clus2, n_traj, n_features, n_pts)
            call get_density_from_cluster(&
                &density2, dataset, feature_vars, coords2, clusters, coeff, pbc, clus2-n_traj, n_traj, n_features, n_pts)
            
            density_mid = density1

            do n = 1, n_features
                path_dir(n) = sign(1d0, path_vec(n,k))
                if (path_vec(n,k) .eq. 0) path_dir(n) = 0 
            end do
            path_coords = coords1

            at_coords2 = .false.
            count = 0
            use1 = .true.

            do while (.not. at_coords2)
                count = count+1
                !if (count .eq. 100) return
                at_coords2 = .true.
                dist = n_features

                do n = 1, n_features
                    if (path_dir(n) .ne. 0) then
                        path_coords(n) = path_coords(n) + path_dir(n)
                        if ((path_coords(n) < 0) .or. (path_coords(n) > n_pts - 1)) then
                            use1 = .false.
                            call modri(path_coords(n), n_pts)
                            goto 200
                        end if
                        
                        if (use1) then
                            call distance_from_line(temp, path_coords, coords1, path_vec(:,k), n_features)
                        else
                            call distance_from_line(temp, path_coords, coords2, path_vec(:,k), n_features)
                        end if

                        write(*,*) temp
                        if (temp .lt. dist) then
                            dist = temp
                            temp_coords = path_coords
                        end if
                        path_coords(n) = path_coords(n) - path_dir(n)
                        call modri(path_coords(n), n_pts)
                    end if
                end do

                path_coords = temp_coords
200             do n = 1, n_features
                    at_coords2 = (at_coords2 .and. (nint(path_coords(n)) .eq. nint(coords2(n))))
                end do

                call get_density_by_clusters(contributions, dataset, feature_vars, path_coords, clusters, &
                & coeff, pbc, n_traj, n_features, n_clus, n_pts)

                temp = 0
                call get_density_from_cluster(&
                &temp, dataset, feature_vars, path_coords, clusters, coeff, pbc, clus1, n_traj, n_features, n_pts)
                call get_density_from_cluster(&
                &temp, dataset, feature_vars, path_coords, clusters, coeff, pbc, clus2, n_traj, n_features, n_pts)

                !temp = contributions(clus1+1) + contributions(clus2+1)
                !do j = 1, n_clus
                !    if (temp .lt. contributions(j)) cycle pair_loop
                !end do

                if (temp .lt. density_mid) density_mid = temp
            end do

            if (density1 .gt. density2) then
                temp = density2
            else
                temp = density1
            end if
            temp = temp - density_mid
            temp = temp/density_mid
            if (temp .lt. metric) metric = temp
        end if

    end do pair_loop

end subroutine compute_merge_metric

subroutine get_clusters(maxima, dataset, feature_vars, coeff, pbc, n_pts, n_traj, n_features)
    implicit none
    
    integer*4, intent(in) :: n_pts, n_traj, n_features, pbc(n_features)
    integer*4, intent(out) :: maxima(n_features, n_traj)
!f2py intent(out) :: maxima

    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_features)
    real*8, intent(in) :: coeff

    real*8 coords(n_features), ocoords(n_features), temp_coords(n_features), density, temp
    integer i, m, n, count
    logical cond

    do i = 1, n_traj
        count = 1
        do n = 1, n_features
            coords(n) = nint(dataset(n,i))
        end do
        temp_coords = coords

        cond = .true.
        do while (cond)
            cond = .false.
            
            density = 0
            call get_density(density, dataset, feature_vars, coords, coeff, pbc, n_traj, n_features, n_pts)

            do n = 1, n_features
                if (pbc(n) > 0) then
                    coords(n) = coords(n) - 1
                    call modri(coords(n), n_pts)
                else if (coords(n) > 0) then
                    coords(n) = coords(n) - 1
                else 
                    goto 100
                end if
                    
                temp = 0
                call get_density(temp, dataset, feature_vars, coords, coeff, pbc, n_traj, n_features, n_pts)
                
                if (temp .gt. density) then
                    density = temp
                    cond = .true.
                    temp_coords = coords
                end if

                coords(n) = coords(n) + 1
                call modri(coords(n), n_pts)

100             if (pbc(n) > 0) then
                    coords(n) = coords(n) + 1
                    call modri(coords(n), n_pts)
                else if (coords(n) < n_pts - 1) then
                    coords(n) = coords(n) + 1
                else 
                    cycle
                end if

                temp = 0
                call get_density(temp, dataset, feature_vars, coords, coeff, pbc, n_traj, n_features, n_pts)

                if (temp .gt. density) then
                    density = temp
                    cond = .true.
                    temp_coords = coords
                end if
                
                coords(n) = coords(n) - 1
                call modri(coords(n), n_pts)
            end do
            
            coords = temp_coords
            count = count + 1
        end do

        maxima(:,i) = coords

    end do
    
end subroutine get_clusters

subroutine generate_grid_2d(grid, dataset, feature_vars, coeff, pbc, n_pts, n_traj)
    implicit none
    
    integer*4, intent(in) :: n_pts, n_traj, pbc(2)
    real*8, intent(out) :: grid(n_pts, n_pts)
!f2py intent(out) :: grid
    real*8, intent(in) :: dataset(2, n_traj), feature_vars(2)
    real*8, intent(in) :: coeff

    integer m, n
    real*8 fs(2)
    
    grid = 0
    do n = 1, n_pts
        do m = 1, n_pts
            fs = [m-1, n-1]
            call get_density(grid(m,n), dataset, feature_vars, fs, coeff, pbc, n_traj, 2, n_pts)
        end do
    end do
end subroutine generate_grid_2d

!subroutine get_gradient(gradient, coords, dataset, feature_vars, coeff, n_traj, n_features)
!    implicit none
!    
!    integer*4, intent(in) :: n_traj, n_features
!
!    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_features)
!    real*8, intent(in) :: coeff
!
!    real*8, intent(in) :: coords(n_features)
!    real*8, intent(out) :: gradient(n_features)
!
!    real*8 density
!    integer i, n
!
!    gradient = 0
!    do i = 1, n_traj
!        density = 0
!        call get_density_from_traj(density, dataset(:,i), feature_vars, coords, coeff, n_features)
!        do n = 1, n_features
!            gradient(n) = gradient(n) - 2*(coords(n) - dataset(n,i))/feature_vars(n) * density
!        end do
!    end do
!end subroutine get_gradient
!
!subroutine get_hessian(hessian, coords, dataset, feature_vars, coeff, n_traj, n_features)
!    implicit none
!    
!    integer*4, intent(in) :: n_traj, n_features
!
!    real*8, intent(in) :: dataset(n_features, n_traj), feature_vars(n_features)
!    real*8, intent(in) :: coeff
!
!    real*8, intent(in) :: coords(n_features)
!    real*8, intent(out) :: hessian(n_features, n_features)
!
!    real*8 density, temp
!    integer i, m, n
!
!    do i = 1, n_traj
!        density = 0
!        call get_density_from_traj(density, dataset(:,i), feature_vars, coords, coeff, n_features)
!        do m = 1, n_features
!            do n = 1, m-1
!                hessian(n, m) = hessian(n, m) + 4*(coords(n) - dataset(n,i))*(coords(m) - dataset(m,i)) &
!                    & /(feature_vars(n)*feature_vars(m)) * density
!                hessian(m,n) = hessian(n,m)    
!            end do
!            
!            temp = coords(m) - dataset(m,i)
!            hessian(m,m) = 2*density/feature_vars(m) * (2*temp*temp/feature_vars(m) - 1)
!        end do
!    end do
!end subroutine get_hessian
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