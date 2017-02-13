% This algorithm is for a Multi-Model SVD model using SGD By WANLU SHI from CISL,FDU

% load the ratings, vieweds and wants data for training procedure
ratings_train = load('ratings_train_1.txt');
% viewed_train = load('views.txt');
% want_train = load('wants.txt');
data_train = load('rvwplusnotwantstrain_1.txt');
% corresponding size of each matrix
rNum_train = size(ratings_train,1);
% vNum = size(viewed_train,1);
% wNum = size(want_train,1);
trainNum = size(data_train,1);

% load the ratings data for testing procedure
ratings_test = load('ratings_test_1.txt');
% corresponding size of rating matrix in testing procedure
rNum_test = size(ratings_test,1);

% load the users and movies
users = load('users.txt');
movies = load('movies.txt');
% users and movies size in training procedure
uNum = size(users,1);
mNum = size(movies,1);

% initialize the max iteration times, factor numbers, learning rate, penalty factor
% and user/item feature matrix
% three model(rating model, viewed model, want model) only share the user features
iter_max = 2000;
K = 10;
learn_rate = 0.01;
lamda1 = 0.05;
lamda2 = 0.1;
lamda3 = 0.1;
P = normrnd(0.0,0.1,K,uNum);
QR = normrnd(0.0,0.1,K,mNum);
QV = normrnd(0.0,0.1,K,mNum);
QW = normrnd(0.0,0.1,K,mNum);
bu = normrnd(0.0,0.1,1,uNum);
bi = normrnd(0.0,0.1,1,mNum);
eta_v = normrnd(0.0,0.1,1,uNum);
eta_w = normrnd(0.0,0.1,1,uNum);
P_final = P;
QR_final = QR;
QV_final = QV;
QW_final = QW;
bu_final = bu;
bi_final = bi;
etav_final = eta_v;
etaw_final = eta_w;

% initialize the weight of R,V,W in cost function
beta_r = 1;
beta_v = 0.1;
beta_w = 0.01;

% compute the global average rating
mu_sum = 0.0;
for i = 1 : rNum_train
	mu_sum = mu_sum + ratings_train(i,3);
end
mu = mu_sum/rNum_train;

% initialize rmse_rate_old
% rmse_rate_old is initialized with a large value
rmse_rate_old = 1000.0;

% parameter estimation in training procedure
it = 1;
while (it<=iter_max)

	% training using rating data
	for tpair = 1 : trainNum
		% record user , movie and rating information
		userIndex = data_train(tpair,1);
		movieIndex = data_train(tpair,2);
        rating = data_train(tpair,3);
        
        if rating >0
			% prediction rating before training
			prediction = mu + bu(1,userIndex) + bi(1,movieIndex) + (P(:,userIndex)'*QR(:,movieIndex)) + eta_v(1,userIndex)*(P(:,userIndex)'*QV(:,movieIndex)) + eta_w(1,userIndex)*(P(:,userIndex)'*QW(:,movieIndex));


			delta_bu = beta_r * (prediction - rating) + lamda2 * bu(1,userIndex);		
			delta_bi = beta_r * (prediction - rating) + lamda2 * bi(1,movieIndex);
			

			% compute the gradient of all features
			delta_u = beta_r * (prediction - rating) * (QR(:,movieIndex) + eta_v(1,userIndex)*QV(:,movieIndex) + eta_w(1,userIndex)*QW(:,movieIndex)) + lamda1 * P(:,userIndex);
			delta_r = beta_r * (prediction - rating) * P(:,userIndex) + lamda2 * QR(:,movieIndex);
			delta_v = beta_r * (prediction - rating) * eta_v(1,userIndex) * P(:,userIndex) + lamda3 * QV(:,movieIndex);
			delta_w = beta_r * (prediction - rating) * eta_w(1,userIndex) * P(:,userIndex) + lamda3 * QW(:,movieIndex);
			delta_etav = beta_r * (prediction - rating) * (P(:,userIndex)'*QV(:,movieIndex)) + lamda2 * eta_v(1,userIndex);
			delta_etaw = beta_r * (prediction - rating) * (P(:,userIndex)'*QW(:,movieIndex)) + lamda2 * eta_w(1,userIndex);

			% update all features	
			bu(1,userIndex) = bu(1,userIndex) - learn_rate * delta_bu;
			bi(1,movieIndex) = bi(1,movieIndex) - learn_rate * delta_bi; 
			P(:,userIndex) = P(:,userIndex) - learn_rate * delta_u; 
			QR(:,movieIndex) = QR(:,movieIndex) - learn_rate * delta_r;		
			QV(:,movieIndex) = QV(:,movieIndex) - learn_rate * delta_v;
			QW(:,movieIndex) = QW(:,movieIndex) - learn_rate * delta_w;	
			eta_v(1,userIndex) = eta_v(1,userIndex) - learn_rate * delta_etav;
			eta_w(1,userIndex) = eta_w(1,userIndex) - learn_rate * delta_etaw;

        elseif rating == -3			

			% prediction viewed preference before training
			prediction = P(:,userIndex)'*QV(:,movieIndex);

			% compute all features
			delta_u = beta_v * (prediction - 1) * QV(:,movieIndex) + lamda1 * P(:,userIndex);
			delta_v = beta_v * (prediction - 1) * P(:,userIndex) + lamda3 * QV(:,movieIndex);

			% update all features
			P(:,userIndex) = P(:,userIndex) - learn_rate * delta_u;
			QV(:,movieIndex) = QV(:,movieIndex) - learn_rate * delta_v;

        else
    
            if rating == -1
                rating = 1;
            else
                rating = 0;
            end
			% prediction want preference before training
			prediction = P(:,userIndex)'*QW(:,movieIndex);

			% compute all features
			delta_u = beta_w * (prediction - rating) * QW(:,movieIndex) + lamda1 * P(:,userIndex);
			delta_w = beta_w * (prediction - rating) * P(:,userIndex) + lamda3 * QW(:,movieIndex);

			% update all features
			P(:,userIndex) = P(:,userIndex) - learn_rate * delta_u;
			QW(:,movieIndex) = QW(:,movieIndex) - learn_rate * delta_w;

        end

	end	

	disp(['finish ', num2str(it),' iteration']);

    % compute new rmse_rate
	rvsum = 0.0;
	for vi = 1 : rNum_train
		ruvali = ratings_train(vi,1);
		rmvali = ratings_train(vi,2);
        ratingvali = ratings_train(vi,3);
	    prediction_v = mu + bu(1,ruvali) + bi(1,rmvali) + (P(:,ruvali)'*QR(:,rmvali)) + eta_v(1,ruvali) * (P(:,ruvali)'*QV(:,rmvali)) + eta_w(1,ruvali) * (P(:,ruvali)'*QW(:,rmvali)); 
        rerr = prediction_v-ratingvali;
		rvsum = rvsum + rerr*rerr;
	end
	rmse_rate_new = sqrt(rvsum/rNum_train);
	disp(['RMSE of Ratings with train set = ', num2str(rmse_rate_new)]);
    
% 	% compute new rmse_rate
% 	rvsum = 0.0;
% 	for vi = 1 : rNum_test
% 		ruvali = ratings_test(vi,1);
% 		rmvali = ratings_test(vi,2);
%         ratingvali = ratings_test(vi,3);
% 	    prediction_v = mu + bu(1,ruvali) + bi(1,rmvali) + (P(:,ruvali)'*QR(:,rmvali)) + eta_v(1,ruvali) * (P(:,ruvali)'*QV(:,rmvali)) + eta_w(1,ruvali) * (P(:,ruvali)'*QW(:,rmvali)); 
%         rerr = prediction_v-ratingvali;
% 		rvsum = rvsum + rerr*rerr;
% 	end
% 	rmse_rate_new = sqrt(rvsum/rNum_test);
% 	disp(['RMSE of Ratings with test set = ', num2str(rmse_rate_new)]);

	% judge convergence condition
    if (rmse_rate_new<=rmse_rate_old) && (abs(rmse_rate_new-rmse_rate_old)>=0.0001)
		P_final = P;
		QR_final = QR;
		QV_final = QV;
		QW_final = QW;
		bu_final = bu;
		bi_final = bi;
		etav_final = eta_v;
		etaw_final = eta_w;

		% set new rmse_rate_old
		rmse_rate_old = rmse_rate_new;
	else
		break;
    end

	it = it + 1;
end


% evaluation using test data by RMSE
rsum = 0.0;
for x = 1 : rNum_test
	rutest = ratings_test(x,1);
	rmtest = ratings_test(x,2);
    ratingtest = ratings_test(x,3);
    prediction = mu + bu_final(1,rutest) + bi_final(1,rmtest) + (P_final(:,rutest)'*QR_final(:,rmtest)) + etav_final(1,rutest) * (P_final(:,rutest)'*QV_final(:,rmtest)) + etaw_final(1,rutest) * (P_final(:,rutest)'*QW_final(:,rmtest)); 
    if prediction > 5.0
        prediction=5.0;
    elseif prediction < 1.0
        prediction=1.0;
    end
    rerr = prediction-ratingtest;
	rsum = rsum + rerr*rerr;
end
rmse_rate = sqrt(rsum/rNum_test);
disp(['RMSE of Ratings in test procedure is ', num2str(rmse_rate)]);