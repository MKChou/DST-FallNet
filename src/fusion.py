def calculate_belief_mass(score):
    return {
        'Fall': score,
        'Normal': 1 - score
    }

def calculate_conflict_coefficient(m_v, m_a):
    K = m_v['Fall'] * m_a['Normal'] + m_v['Normal'] * m_a['Fall']
    return K

def dempster_rule(m_v, m_a):
    K = calculate_conflict_coefficient(m_v, m_a)
    if K == 1:
        return {'Fall': 0.5, 'Normal': 0.5}
    m_f = {
        'Fall': (m_v['Fall'] * m_a['Fall']) / (1 - K),
        'Normal': (m_v['Normal'] * m_a['Normal']) / (1 - K)
    }
    return m_f 