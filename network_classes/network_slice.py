class Slice:
    def __init__(self, UEs, b_width, r_sla=0.1):
        self.UEs = UEs
        self.b_width = b_width
        self.r_sla = r_sla  # guaranteed date rate of the slice

    def set_sla(self, new_sla):
        self.r_sla = new_sla
