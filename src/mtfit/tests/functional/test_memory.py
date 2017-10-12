class _MemoryTestCase(unittest.TestCase):
    def mem_data(self,location_samples=False):
        from .mem_report import size_report
        data={'PPolarity':{'Stations':{'Azimuth':np.array([90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349,90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349,90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349,90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349]),
                                        'TakeOffAngle':np.array([30.0,60.0,40.,23,12,55,12,32,78,44,30.0,60.0,40.,23,12,55,12,32,78,44,30.0,60.0,40.,23,12,55,12,32,78,44,30.0,60.0,40.,23,12,55,12,32,78,44]),
                                        'Name':['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31','S32','S33','S34','S35','S36','S37','S38','S39','S40']},
            'Measured':np.matrix([[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1],[1],[-1]]),
            'Error':np.matrix([[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4],[ 0.4]])}}

        data2={'P/SHRMSAmplitudeRatio':{'Stations':{'Azimuth':np.array([90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349,90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349,90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349,90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349]),
                                        'TakeOffAngle':np.array([30.0,60.0,40.,23,12,55,12,32,78,44,30.0,60.0,40.,23,12,55,12,32,78,44,30.0,60.0,40.,23,12,55,12,32,78,44,30.0,60.0,40.,23,12,55,12,32,78,44]),
                                        'Name':['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31','S32','S33','S34','S35','S36','S37','S38','S39','S40']},
            'Measured':np.matrix([[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1],[ 1.3664,1],[1.0038,1]]),
            'Error':np.matrix([[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7],[ 0.7,0.7]])}}
        
        a_polarity,error_polarity,incorrect_polarity_prob=polarity_matrix(data,location_samples)
        if type(location_samples)==bool:
            n_location_samples=1
        else:
            n_location_samples=len(location_samples)
        a_polarity_prob,polarity_prob,incorrect_polarity_p_prob=polarity_prob_matrix(data,location_samples)
        a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=amplitude_ratio_matrix(data2,location_samples)
        return  a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio
    def location_samples(self,n=1000):
        print str(n)+' samples'
        loc_sample={'Azimuth':np.array([90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349,90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349,90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349,90.0,270.0,140.0,133.2,24.5,22.1,123.2,195,229,349]),
                                        'TakeOffAngle':np.array([30.0,60.0,40.,23,12,55,12,32,78,44,30.0,60.0,40.,23,12,55,12,32,78,44,30.0,60.0,40.,23,12,55,12,32,78,44,30.0,60.0,40.,23,12,55,12,32,78,44]),
                                        'Name':['S1','S2','S3','S4','S5','S6','S7','S8','S9','S10','S11','S12','S13','S14','S15','S16','S17','S18','S19','S20','S21','S22','S23','S24','S25','S26','S27','S28','S29','S30','S31','S32','S33','S34','S35','S36','S37','S38','S39','S40']}
        loc_samples=[]
        i=0
        while i<n:
            loc_samples.append(loc_sample)
            i+=1
        return loc_samples
    def test_data_initialisation(self):
        try:
            from .mem_report import size_report
        except:
            from mem_report import size_report

        size_report(float=1.0,np_matrix=np.matrix([[1.0]]))
        a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data()
        total,estimated=size_report(1,40,a_polarity=a_polarity,error_polarity=error_polarity,incorrect_polarity_prob=incorrect_polarity_prob,a1_amplitude_ratio=a1_amplitude_ratio,a2_amplitude_ratio=a2_amplitude_ratio,amplitude_ratio=amplitude_ratio,percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio)
        self.assertTrue(float(total)/estimated<=1.0)
        gc.collect()
        loc_samples=self.location_samples(2)
        a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data(loc_samples)
        total,estimated=size_report(len(loc_samples),40,a_polarity=a_polarity,error_polarity=error_polarity,incorrect_polarity_prob=incorrect_polarity_prob,a1_amplitude_ratio=a1_amplitude_ratio,a2_amplitude_ratio=a2_amplitude_ratio,amplitude_ratio=amplitude_ratio,percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio)
        self.assertTrue(float(total)/estimated<=1.0)
        del loc_samples
        gc.collect()
        loc_samples=self.location_samples(20)
        a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data(loc_samples)
        total,estimated=size_report(len(loc_samples),40,a_polarity=a_polarity,error_polarity=error_polarity,incorrect_polarity_prob=incorrect_polarity_prob,a1_amplitude_ratio=a1_amplitude_ratio,a2_amplitude_ratio=a2_amplitude_ratio,amplitude_ratio=amplitude_ratio,percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio)
        self.assertTrue(float(total)/estimated<=1.0)
        del loc_samples
        gc.collect()
        loc_samples=self.location_samples(100)
        a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data(loc_samples)
        total,estimated=size_report(len(loc_samples),40,a_polarity=a_polarity,error_polarity=error_polarity,incorrect_polarity_prob=incorrect_polarity_prob,a1_amplitude_ratio=a1_amplitude_ratio,a2_amplitude_ratio=a2_amplitude_ratio,amplitude_ratio=amplitude_ratio,percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio)
        self.assertTrue(float(total)/estimated<=1.0)
        del loc_samples
        gc.collect()
        loc_samples=self.location_samples(200)
        a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data(loc_samples)
        total,estimated=size_report(len(loc_samples),40,a_polarity=a_polarity,error_polarity=error_polarity,incorrect_polarity_prob=incorrect_polarity_prob,a1_amplitude_ratio=a1_amplitude_ratio,a2_amplitude_ratio=a2_amplitude_ratio,amplitude_ratio=amplitude_ratio,percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio)
        self.assertTrue(float(total)/estimated<=1.0)
        del loc_samples
        gc.collect()
        loc_samples=self.location_samples(500)
        a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data(loc_samples)
        total,estimated=size_report(len(loc_samples),40,a_polarity=a_polarity,error_polarity=error_polarity,incorrect_polarity_prob=incorrect_polarity_prob,a1_amplitude_ratio=a1_amplitude_ratio,a2_amplitude_ratio=a2_amplitude_ratio,amplitude_ratio=amplitude_ratio,percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio)
        self.assertTrue(float(total)/estimated<=1.0)
        del loc_samples
        gc.collect()
        loc_samples=self.location_samples(1000)
        a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data(loc_samples)
        total,estimated=size_report(len(loc_samples),40,a_polarity=a_polarity,error_polarity=error_polarity,incorrect_polarity_prob=incorrect_polarity_prob,a1_amplitude_ratio=a1_amplitude_ratio,a2_amplitude_ratio=a2_amplitude_ratio,amplitude_ratio=amplitude_ratio,percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio)
        self.assertTrue(float(total)/estimated<=1.0)
        del loc_samples
        gc.collect()
        loc_samples=self.location_samples(2000)
        a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data(loc_samples)
        total,estimated=size_report(len(loc_samples),40,a_polarity=a_polarity,error_polarity=error_polarity,incorrect_polarity_prob=incorrect_polarity_prob,a1_amplitude_ratio=a1_amplitude_ratio,a2_amplitude_ratio=a2_amplitude_ratio,amplitude_ratio=amplitude_ratio,percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio)
        self.assertTrue(float(total)/estimated<=1.0)
        del loc_samples
        gc.collect()
        loc_samples=self.location_samples(10000)
        a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data(loc_samples)
        total,estimated=size_report(len(loc_samples),40,a_polarity=a_polarity,error_polarity=error_polarity,incorrect_polarity_prob=incorrect_polarity_prob,a1_amplitude_ratio=a1_amplitude_ratio,a2_amplitude_ratio=a2_amplitude_ratio,amplitude_ratio=amplitude_ratio,percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio)
        self.assertTrue(float(total)/estimated<=1.0)
        del loc_samples
        gc.collect()
        loc_samples=self.location_samples(20000)
        a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data(loc_samples)
        total,estimated=size_report(len(loc_samples),40,a_polarity=a_polarity,error_polarity=error_polarity,incorrect_polarity_prob=incorrect_polarity_prob,a1_amplitude_ratio=a1_amplitude_ratio,a2_amplitude_ratio=a2_amplitude_ratio,amplitude_ratio=amplitude_ratio,percentage_error1_amplitude_ratio=percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=percentage_error2_amplitude_ratio)
        self.assertTrue(float(total)/estimated<=1.0)
        del loc_samples
        gc.collect()
    def randomMT(self,number_samples):
        M=np.random.randn(6,number_samples)
        X=np.sqrt(np.sum(np.multiply(M,M),axis=0))
        return np.matrix(M/X),X
    def test_polarity_probability_calculations(self):
        try:
            from .mem_report import size_report
        except:
            from mem_report import size_report
        for n_loc_samples in [1,100,500,1000,2000,5000]:
            for n_samples in [1,100,500]:             
                loc_samples=self.location_samples(n_loc_samples)
                print n_loc_samples,' location samples and ',n_samples,' MT samples'
                MT,X=self.randomMT(n_samples)
                total,estimated=size_report(n_samples=n_samples,MT=MT)
                self.assertTrue(float(total)/estimated<=1.0)
                size_report(MTsqrt=X)
                a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data(loc_samples)
                X=np.tensordot(a_polarity,MT,1)
                P,X2,S2,I2=self.polarity_expansion(X,error_polarity,incorrect_polarity_prob)
                total,estimated=size_report(n_loc_samples,40,n_samples,a_polarity=a_polarity,error_polarity=error_polarity,incorrect_polarity_prob=incorrect_polarity_prob,tensordot=X,tensordot_2=X2,expanded_error=S2,expanded_incorrect=I2,probability=P)
                self.assertTrue(float(total)/estimated<=1.0)
                del loc_samples
                gc.collect()
    def polarity_expansion(self,X,sigma,incorrect_polarity_prob):
        import warnings,gc
        import numpy as np
        np.seterr(divide='print',invalid='print')#Set to prevent numpy warnings
        from scipy.stats import norm as normalDist
        from scipy.stats import beta as betaDist
        from scipy.special import erf
        if type(sigma) in [type(np.array([])),type(np.matrix([]))]:
            sigma[sigma==0]=0.000000000000000000000001
        elif sigma==0:
            sigma=0.000000000000000000000001
        if type(X) in [type(np.array([])),type(np.matrix([]))] and X.ndim>2:
            sigma=np.array(sigma)
            if sigma.ndim==2:
                sigma=np.expand_dims(sigma,1)
            if type(incorrect_polarity_prob) in [type(np.array([])),type(np.matrix([]))] and len(incorrect_polarity_prob.shape)==2:
                incorrect_polarity_prob=np.expand_dims(incorrect_polarity_prob,1)
        p= np.multiply(0.5*(1+erf(X/(np.sqrt(2)*sigma))),(1-incorrect_polarity_prob))+np.multiply(0.5*(1+erf(-X/(np.sqrt(2)*sigma))),incorrect_polarity_prob)
        p[np.isnan(p)]=0
        return p,X,sigma,incorrect_polarity_prob
    def test_amp_rat_probability_calculations(self):
        try:
            from .mem_report import size_report
        except:
            from mem_report import size_report
        for n_loc_samples in [1,100,500,1000,2000]:
            for n_samples in [1,100,500]:             
                loc_samples=self.location_samples(n_loc_samples)
                print n_loc_samples,' location samples and ',n_samples,' MT samples'
                MT,X=self.randomMT(n_samples)
                total,estimated=size_report(n_samples=n_samples,MT=MT)
                self.assertTrue(float(total)/estimated<=1.0)
                size_report(MTsqrt=X)
                a_polarity,error_polarity,incorrect_polarity_prob,a_polarity_prob,polarity_prob,incorrect_polarity_p_prob,a1_amplitude_ratio,a2_amplitude_ratio,amplitude_ratio,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio=self.mem_data(loc_samples)
                modelled_numerator=np.abs(np.tensordot(a1_amplitude_ratio,MT,1))
                modelled_denominator=np.abs(np.tensordot(a2_amplitude_ratio,MT,1))
                P,zz,sxsx,sxsy,sysy,muxmux,a,b,c,d,mux,nuy,sx,sy,percentage_error_numerator,percentage_error_denominator=self.amplitude_ratio_expansion(amplitude_ratio,modelled_numerator,modelled_denominator,percentage_error1_amplitude_ratio,percentage_error2_amplitude_ratio)
                total,estimated=size_report(n_loc_samples,40,n_samples,tensordot=modelled_numerator,tensordot_d=modelled_denominator,tensordot_2=mux,tensordot_d_2=muy,expanded_error=percentage_error_numerator,expanded_error2=sx,
                    expanded_error_d=percentage_error_denominator,expanded_error_d_2=sy,expanded_zz=zz,expanded_sxsx=sxsx,expanded_sxsy=sxsy,expanded_sysy=sysy,expanded_muxmux=muxmux,expanded_a=a,expanded_b=b,expanded_c=c,expanded_d=d,probability=p)
                self.assertTrue(float(total)/estimated<=1.0)
                del loc_samples
                gc.collect()
    def amplitude_ratio_expansion(self,ratio,mux,muy,percentage_error_numerator,percentage_error_denominator):
        if type(mux) in [type(np.array([])),type(np.matrix([]))] and len(mux.shape)==3:
            ratio=np.expand_dims(ratio,1)
            if len(percentage_error_numerator.shape)==2:
                percentage_error_numerator=np.expand_dims(percentage_error_numerator,1)
            if len(percentage_error_denominator.shape)==2:
                percentage_error_denominator=np.expand_dims(percentage_error_denominator,1)
        numerator_error=np.multiply(percentage_error_numerator,mux)
        denominator_error=np.multiply(percentage_error_denominator,muy)
        p,zz,sxsx,sxsy,sysy,muxmux,a,b,c,d,mux,nuy,sx,sy= self.ratio_pdf_expansion(ratio,mux,muy,numerator_error,denominator_error)+ratio_pdf(-ratio,mux,muy,numerator_error,denominator_error)
        return p,zz,sxsx,sxsy,sysy,muxmux,a,b,c,d,mux,nuy,sx,sy,percentage_error_numerator,percentage_error_denominator
    def ratio_pdf_expansion(self,z,mux,muy,sx,sy,corr=0):
        from probability import gaussian_cdf
        np.seterr(divide='ignore',invalid='ignore')#Set to prevent numpy warnings
        if type(z) in [type(np.array([])),type(np.matrix([]))] and len(z.shape)==3:
            if type(mux) in [type(np.array([])),type(np.matrix([]))] and len(mux.shape)==2:
                mux=np.expand_dims(mux,1)
            if type(muy) in [type(np.array([])),type(np.matrix([]))] and len(muy.shape)==2:
                muy=np.expand_dims(muy,1)
            if len(sx.shape)==2:
                sx=np.expand_dims(sx,1)
            if len(sy.shape)==2:
                sy=np.expand_dims(sy,1)
        zz=np.multiply(z,z)
        sxsx=np.multiply(sx,sx)
        sxsy=np.multiply(sx,sy)
        sysy=np.multiply(sy,sy)
        muxmux=np.multiply(mux,mux)
        if corr>0:
            a=np.sqrt(np.divide(zz,sxsx)-2*corr*np.divide(z,sxsy)+1/sysy)
        else:        
            a=np.sqrt(np.divide(zz,sxsx)+1/sysy)
        b=np.divide(np.multiply(mux,z),sxsx)-(corr*np.divide(sxsx,sxsy))+np.divide(muy,sysy)
        c=np.divide(muxmux,sxsx)+np.divide(np.multiply(muy,muy),sysy)
        if corr>0:
            c-=(2*corr*np.divide(np.multiply(mux,muy),sxsy))
        
        d=np.exp(np.divide((np.multiply(b,b)-np.multiply(c,np.multiply(a,a))),(2*(1-corr*corr)*np.multiply(a,a))))
        p= np.divide(np.multiply(b,d),(np.sqrt(2*np.pi)*np.multiply(sx,np.multiply(sy,np.multiply(a,np.multiply(a,a))))))
        p=np.multiply(p,(gaussian_cdf(np.divide(b,(np.sqrt(1-corr*corr)*a)),0,1)-gaussian_cdf(np.divide(-b,(np.sqrt(1-corr*corr)*a)),0,1)))
        p+=np.multiply((np.sqrt(1-corr*corr)/(np.pi*np.multiply(sx,np.multiply(sy,np.multiply(a,a))))),np.exp(-c/(2*(1-corr*corr))))
        if type(p) in [type(np.array([])),type(np.matrix([]))] :
            p[np.isnan(p)]=0
        return p,zz,sxsx,sxsy,sysy,muxmux,a,b,c,d,mux,nuy,sx,sy
def _memory_test_suite():
    tests=[]
    tests.append(unittest.TestLoader().loadTestsFromTestCase(__MemoryTestCase))
    return unittest.TestSuite(tests)
